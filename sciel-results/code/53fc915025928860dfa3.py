import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s/150s kill

    n = 40

    def would_create_c6(adj, u, v):
        """
        Check if adding edge (u, v) to the graph represented by adjacency bitmasks `adj`
        creates a C6.
        A new C6 must use the new edge (u, v).
        So we need a path of length 5 from u to v in the current graph:
        u - a - b - c - d - v, with a, b, c, d distinct and not u, v.
        """
        Nu = adj[u]
        Nv = adj[v]

        # a in N(u) \ {v}
        # d in N(v) \ {u}
        # a != d

        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a

            # d in N(v) \ {u, a}
            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d

                Na = adj[a]
                Nd = adj[d]

                # b in N(a) \ {u, v, d}
                # c in N(d) \ {u, v, a}
                # such that b != c and (b, c) is an edge.
                
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)
                
                # We need b in b_mask such that adj[b] & c_mask != 0
                # Note: if b == c, adj[b] doesn't have bit b, so adj[b] & c_mask won't have bit b.
                # But c_mask might have bit b if b is in N(d) and b != a, u, v.
                # If b is in c_mask and we pick b, then adj[b] & c_mask checks if b has any neighbor in c_mask.
                # If the only neighbor in c_mask is b itself? No, adj[b] doesn't have b.
                # So if adj[b] & c_mask != 0, there is some c in c_mask adjacent to b.
                # Is c distinct from b? Yes, because adj[b] doesn't have bit b.
                # Is c distinct from a, d, u, v? Yes, because c is in c_mask which excludes them.
                # Is b distinct from a, d, u, v? Yes, because b is in b_mask which excludes them.
                # Are b and c distinct? Yes.
                
                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    
                    if adj[b] & c_mask:
                        return True
        
        return False

    def run_greedy(seed_val, init_type='k5'):
        random.seed(seed_val)
        
        adj = [0] * n
        edges = set()
        
        if init_type == 'k5':
            # Start with 8 K5s
            for k in range(8):
                for a in range(5):
                    for b in range(a + 1, 5):
                        u, v = 5 * k + a, 5 * k + b
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edges.add((u, v))
        elif init_type == 'empty':
            pass
        elif init_type == 'random':
            # Random sparse start
            num_init = random.randint(10, 30)
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            for u, v in all_pairs[:num_init]:
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.add((u, v))
        
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if time.time() > deadline:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        
        return edges

    def local_search(edges_set, seed_val):
        random.seed(seed_val)
        
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        current_edges = set(edges_set)
        
        # Generate non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    non_edges.append((i, j))
        
        if not non_edges:
            return current_edges

        random.shuffle(non_edges)
        
        # Pass 1: Add any possible non-edge
        improved = False
        for u, v in non_edges:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)): # Check if still non-edge (might have been added)
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    improved = True
        
        if improved:
            return current_edges

        # Pass 2: Try swaps (remove 1, add 2)
        # We want to find an edge e and two non-edges e1, e2 such that
        # G - e + e1 + e2 is C6-free.
        # Heuristic:
        # 1. Pick a random edge to remove.
        # 2. Try to add two non-edges that are "near" the removed edge or random.
        
        removable = list(current_edges)
        random.shuffle(removable)
        
        # To speed up, we don't try all swaps. We try a limited number.
        num_swaps_to_try = 20
        
        for _ in range(num_swaps_to_try):
            if time.time() > deadline - 5:
                break
            
            if not removable:
                break
                
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            
            # Save state
            old_adj = adj[:]
            old_edges = set(current_edges)
            
            # Remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            
            # Try to add two non-edges
            added = 0
            # We prioritize non-edges that are incident to u_rem or v_rem, or just random
            # Let's just pick from the shuffled non_edges list
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 5:
                    break
                if (u, v) in current_edges:
                    continue
                # Check if u,v is still a non-edge in the modified graph
                # Since we only removed one edge, and (u,v) was a non-edge before,
                # it's still a non-edge unless (u,v) == (u_rem, v_rem) which we skipped.
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            
            if added == 2:
                # We gained 1 edge
                return current_edges
            else:
                # Revert
                adj = old_adj
                current_edges = old_edges

        return current_edges

    best_edges = set()
    best_count = 0
    
    # Strategy:
    # 1. Run a few K5-based greedies.
    # 2. Run a few empty-based greedies.
    # 3. Run a few random-based greedies.
    # 4. Apply local search to the best ones.
    
    seed_counter = 42
    
    # Phase 1: Diverse Greedies
    strategies = []
    for i in range(5):
        strategies.append(('k5', 42 + i))
    for i in range(5):
        strategies.append(('empty', 100 + i))
    for i in range(5):
        strategies.append(('random', 200 + i))
    
    # Shuffle strategies to interleave
    random.seed(0)
    random.shuffle(strategies)
    
    candidates = []
    
    for init_type, seed in strategies:
        if time.time() > deadline - 10:
            break
        edges = run_greedy(seed, init_type)
        candidates.append(edges)
    
    # Sort candidates by size
    candidates.sort(key=len, reverse=True)
    
    # Phase 2: Local Search on top candidates
    # We only have time for a few local searches
    max_ls = 3
    for i in range(min(max_ls, len(candidates))):
        if time.time() > deadline - 5:
            break
        edges = local_search(candidates[i], 1000 + i)
        if len(edges) > len(best_edges):
            best_edges = edges
            best_count = len(edges)
    
    # Also check if any greedy was better than LS results (unlikely but possible if LS failed)
    for edges in candidates:
        if len(edges) > len(best_edges):
            best_edges = edges
            best_count = len(edges)
            
    # Convert set of tuples to list of lists
    return [list(e) for e in best_edges]
