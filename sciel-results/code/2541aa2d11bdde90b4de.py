import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill
    n = 20
    best_edges = 0
    best_adj = None

    # Precompute all triples
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def edge_count(adj):
        return sum(bin(x).count('1') for x in adj) // 2
    
    def get_edges(adj):
        return [[u, v] for u in range(n) for v in range(u+1, n) if (adj[u] >> v) & 1]
    
    def is_k33_fast(adj):
        """Check if graph is K3,3-free (no three vertices have 3+ common neighbors)"""
        for (i, j, k) in triples:
            common = adj[i] & adj[j] & adj[k]
            # Check if popcount >= 3
            if common & (common - 1) and (common & (common - 1) & (common & (common - 1) - 1) or bin(common).count('1') >= 3):
                # More efficient: just use bin count for clarity, it's fast enough
                if bin(common).count('1') >= 3:
                    return False
        return True
    
    def can_add_edge(adj, u, v):
        """Check if adding edge (u,v) would violate K3,3-free condition"""
        # Adding edge (u,v) means u and v become neighbors
        # This could create a K3,3 if there exist i,j such that 
        # {u,v} along with some other vertex form one partition side,
        # and 3 vertices form the other
        # 
        # More precisely: after adding (u,v), for any pair (a,b), 
        # if u and v are both in N(a) ∩ N(b) ∩ N(c) for some c... 
        # 
        # Actually the condition is: no three vertices have 3 common neighbors.
        # Adding (u,v) affects common neighbors of any triple containing u or v.
        # 
        # For triple (u, v, x): common neighbors = N(u) ∩ N(v) ∩ N(x) after adding
        # For triple (u, x, y): common neighbors = N(u) ∩ N(x) ∩ N(y) after adding
        # For triple (x, y, z) not containing u or v: unchanged
        #
        # The key insight: adding (u,v) only creates new common-neighbor relationships
        # involving both u and v simultaneously, or involving one of them with the new edge.
        #
        # Actually, let me think more carefully. The condition "no three vertices have 
        # three common neighbors" means: for all distinct i,j,k, |N(i)∩N(j)∩N(k)| ≤ 2.
        #
        # Adding edge (u,v) changes N(u) to N(u)∪{v} and N(v) to N(v)∪{u}.
        #
        # New triples to check:
        # 1. Triples containing both u and v: (u,v,x) for any x≠u,v
        #    Common neighbors of u,v,x after adding = (N(u)∪{v}) ∩ (N(v)∪{u}) ∩ N(x)
        #    = [(N(u) ∩ N(v) ∩ N(x)) ∪ (N(u) ∩ {u} ∩ N(x)) ∪ ({v} ∩ N(v) ∩ N(x)) ∪ ({v} ∩ {u} ∩ N(x)) ∪ ...]
        #    This gets complicated. Let me just check all triples that include u or v.
        #
        # Simpler approach: just check all triples that include u or v after adding
        # temporarily, then revert.
        
        # Temporarily add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        valid = True
        # Check all triples that include u or v
        for (i, j, k) in triples:
            if i == u or j == u or k == u or i == v or j == v or k == v:
                common = adj[i] & adj[j] & adj[k]
                if bin(common).count('1') >= 3:
                    valid = False
                    break
        
        # Revert
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        return valid
    
    # Greedy construction with multiple restarts
    for restart in range(2000):
        if time.time() > deadline - 5.0:
            break
        
        # Try different strategies
        strategy = restart % 4
        
        adj = [0] * n
        
        if strategy == 0:
            # Random dense initialization
            for u, v in all_edges:
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Purge violations
            for _ in range(500):
                violated = False
                for (i, j, k) in triples:
                    common = adj[i] & adj[j] & adj[k]
                    if bin(common).count('1') >= 3:
                        violated = True
                        # Remove a random edge from the violating structure
                        # Common neighbors have edges to i, j, k
                        # Pick a random common neighbor and remove one edge
                        c = common
                        bits = []
                        while c:
                            lsb = c & -c
                            bits.append(lsb.bit_length() - 1)
                            c -= lsb
                        x = random.choice(bits)
                        # Remove edge between x and a random one of i,j,k
                        choice = random.choice([i, j, k])
                        adj[x] &= ~(1 << choice)
                        adj[choice] &= ~(1 << x)
                        break
                if not violated:
                    break
                    
        elif strategy == 1:
            # Shuffled greedy
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        elif strategy == 2:
            # Start from a structured graph (e.g., random regular-like)
            # Try a random permutation-based construction
            perm = list(range(n))
            random.shuffle(perm)
            # Connect each vertex to next few in permutation
            for i in range(n):
                for d in [1, 2, 3]:
                    j = perm[(i + d) % n]
                    if i != j:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Then try to add more
            shuffled = all_edges[:]
            random.shuffle(shuffled)
            for u, v in shuffled:
                if not (adj[u] >> v) & 1:
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
        else:
            # Sparse random then fill
            for u, v in all_edges:
                if random.random() < 0.2:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Fill greedily
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Local search: try to add any missing edge
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        # Try some swaps to improve
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v) & 1]
            existing = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            
            if not missing or not existing:
                break
            
            # Try adding a missing edge
            random.shuffle(missing)
            added = False
            for u, v in missing[:30]:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if added:
                continue
            
            # Try swap: remove an existing edge, add a missing one
            swap_success = False
            for _ in range(20):
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                
                # Temporarily remove (u1,v1) and add (u2,v2)
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    swap_success = True
                    break
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
            
            if swap_success:
                continue
        
        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]
            
            # If we've found a very dense graph, we can be more aggressive
            if best_edges > 100:
                # Try more restarts with similar density
                pass
    
    # Final: try to polish the best graph
    if best_adj is not None:
        adj = best_adj[:]
        
        # Try to add any missing edges
        improved = True
        while improved:
            improved = False
            for u, v in all_edges:
                if not (adj[u] >> v) & 1:
                    if can_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        # Try swaps
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            
            missing = [(u, v) for u, v in all_edges if not (adj[u] >> v) & 1]
            existing = [(u, v) for u, v in all_edges if (adj[u] >> v) & 1]
            
            if not missing or not existing:
                break
            
            # Try adding a missing edge
            random.shuffle(missing)
            added = False
            for u, v in missing[:50]:
                if can_add_edge(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added = True
                    break
            
            if added:
                ec = edge_count(adj)
                if ec > best_edges:
                    best_edges = ec
                    best_adj = adj[:]
                continue
            
            # Try swap
            for _ in range(50):
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                
                if can_add_edge(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    break
                else:
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
            
            ec = edge_count(adj)
            if ec > best_edges:
                best_edges = ec
                best_adj = adj[:]
    
    if best_adj:
        return get_edges(best_adj)
    else:
        # Fallback: empty graph
        return []
