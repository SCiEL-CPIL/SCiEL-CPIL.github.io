import random
import time
import itertools

def construct():
    start_time = time.time()
    timeout = 115  # Safety margin for 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    triples = [(i, j, k) for i in range(n) for j in range(i+1, n) for k in range(j+1, n)]
    n_triples = len(triples)
    
    def conflict_count(adj):
        """Count number of triples with >= 3 common neighbors."""
        count = 0
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                count += 1
        return count
    
    def is_valid(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return False
        return True
    
    def find_violation(adj):
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                return (i, j, k), common
        return None, None
    
    def count_edges(adj):
        return sum(a.bit_count() for a in adj) // 2
    
    def edge_conflict_freq(adj):
        """Count how many violating triples each edge participates in (as common neighbor link)."""
        # An edge (u,v) contributes to conflicts where u and v are both in a triple 
        # and share >=3 common neighbors. More precisely, for each violating triple (i,j,k),
        # the common neighbors form a K3,3 with {i,j,k}. Each pair among the common neighbors
        # is an "excess" edge. But simpler: for each violating triple, count edges among common neighbors.
        freq = [0] * n
        for i, j, k in triples:
            common = adj[i] & adj[j] & adj[k]
            if common.bit_count() >= 3:
                # Each common neighbor is connected to i, j, k
                # The edges (i,c), (j,c), (k,c) for c in common are "conflicting"
                # Also edges among common neighbors
                c_list = [x for x in range(n) if (common >> x) & 1]
                for c in c_list:
                    freq[i] += 1
                    freq[j] += 1
                    freq[k] += 1
                    freq[c] += 1
                # Edges among common neighbors
                for a in range(len(c_list)):
                    for b in range(a+1, len(c_list)):
                        freq[c_list[a]] += 1
                        freq[c_list[b]] += 1
        return freq
    
    def greedy_add(adj, current_edges):
        """Try to add missing edges greedily."""
        improved = True
        while improved and time.time() - start_time < timeout - 5:
            improved = False
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            random.shuffle(missing)
            
            for u, v in missing:
                if time.time() - start_time > timeout - 5:
                    break
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_valid(adj):
                    current_edges += 1
                    improved = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        return current_edges
    
    def simulated_annealing(adj, current_edges, max_iter=5000):
        """SA with conflict count as smooth surrogate. Accepts worse states with prob exp(-delta/T)."""
        best_local = current_edges
        best_local_adj = [a for a in adj]
        
        if current_edges < 0:
            return best_local, best_local_adj, False
        
        T_start = 5.0
        T_end = 0.1
        # Cool over max_iter steps
        alpha = (T_end / T_start) ** (1.0 / max(1, max_iter))
        
        current_conf = conflict_count(adj)
        
        for it in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
            T = T_start * (alpha ** it)
            
            # Choose a random move: add or remove a random edge
            if random.random() < 0.5:
                # Try to add a missing edge
                missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
                if not missing:
                    break
                u, v = random.choice(missing)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_conf = conflict_count(adj)
                delta_conf = new_conf - current_conf
                delta_edges = 1
                
                # Accept if improves edges and doesn't worsen conflicts too much,
                # or with probability based on temperature
                if delta_edges > 0 and delta_conf <= 0:
                    current_edges += 1
                    current_conf = new_conf
                    if current_edges > best_local:
                        best_local = current_edges
                        best_local_adj = [a for a in adj]
                elif delta_edges > 0:
                    # Accept if probability allows
                    # Penalty for conflicts
                    effective_delta = -delta_edges * 1.0 + delta_conf * 0.5
                    if random.random() < min(1.0, max(0.0, -effective_delta / max(T, 1e-9))):
                        current_edges += 1
                        current_conf = new_conf
                        if current_edges > best_local:
                            best_local = current_edges
                            best_local_adj = [a for a in adj]
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                # Try to remove an existing edge
                existing = [(i, j) for i, j in all_pairs if (adj[i] >> j) & 1]
                if not existing:
                    break
                u, v = random.choice(existing)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_conf = conflict_count(adj)
                delta_conf = new_conf - current_conf
                delta_edges = -1
                
                # Only accept removal if it helps conflicts enough
                if delta_conf < 0:
                    # Accept if conflicts improved
                    effective_delta = -delta_edges * 1.0 + delta_conf * 0.5
                    if random.random() < min(1.0, max(0.0, -effective_delta / max(T, 1e-9))):
                        current_edges -= 1
                        current_conf = new_conf
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Final: try greedy add on best local
        adj2 = [a for a in best_local_adj]
        edges2 = greedy_add(adj2, best_local)
        return edges2, adj2, best_local > current_edges
    
    def two_edge_swap(adj, current_edges, max_attempts=300):
        """Remove 2 edges, add 2 different edges."""
        for _ in range(max_attempts):
            if time.time() - start_time > timeout - 5:
                break
            
            existing = [(i, j) for i, j in all_pairs if (adj[i] >> j) & 1]
            missing = [(i, j) for i, j in all_pairs if not (adj[i] >> j) & 1]
            if len(existing) < 2 or len(missing) < 2:
                break
            
            # Pick 2 edges to remove
            e_rem1, e_rem2 = random.sample(existing, 2)
            u1r, v1r = e_rem1
            u2r, v2r = e_rem2
            
            # Pick 2 edges to add (prefer ones sharing vertices with removed edges)
            removed_verts = {u1r, v1r, u2r, v2r}
            shared_missing = [(i, j) for i, j in missing if i in removed_verts or j in removed_verts]
            if len(shared_missing) >= 2:
                e_add1, e_add2 = random.sample(shared_missing, 2)
            else:
                e_add1, e_add2 = random.sample(missing, 2)
            
            # Apply
            adj[u1r] &= ~(1 << v1r)
            adj[v1r] &= ~(1 << u1r)
            adj[u2r] &= ~(1 << v2r)
            adj[v2r] &= ~(1 << u2r)
            
            u1a, v1a = e_add1
            u2a, v2a = e_add2
            adj[u1a] |= (1 << v1a)
            adj[v1a] |= (1 << u1a)
            adj[u2a] |= (1 << v2a)
            adj[v2a] |= (1 << u2a)
            
            if is_valid(adj):
                # Net change: removed 2, added 2, so edge count same
                # But structure improved - continue
                return True
            else:
                # Revert
                adj[u1a] &= ~(1 << v1a)
                adj[v1a] &= ~(1 << u1a)
                adj[u2a] &= ~(1 << v2a)
                adj[v2a] &= ~(1 << u2a)
                adj[u1r] |= (1 << v1r)
                adj[v1r] |= (1 << u1r)
                adj[u2r] |= (1 << v2r)
                adj[v2r] |= (1 << u2r)
        
        return False
    
    def targeted_remove(adj, current_edges, max_rounds=50):
        """Remove edges that appear in many conflicts, then greedily re-add."""
        for _ in range(max_rounds):
            if time.time() - start_time > timeout - 5:
                break
            if not conflict_count(adj):
                break
            
            freq = edge_conflict_freq(adj)
            # Find high-conflict vertices
            # Remove edges from high-conflict vertices to other high-conflict vertices
            high_conf = [i for i in range(n) if freq[i] > 0]
            if not high_conf:
                break
            
            # Pick a random high-conflict vertex and remove one of its edges
            v = random.choice(high_conf)
            neighbors = [u for u in range(n) if u != v and (adj[v] >> u) & 1]
            if not neighbors:
                break
            
            # Pick neighbor with high conflict freq
            neighbors.sort(key=lambda u: freq[u], reverse=True)
            if random.random() < 0.7:
                u = neighbors[0]
            else:
                u = random.choice(neighbors)
            
            adj[v] &= ~(1 << u)
            adj[u] &= ~(1 << v)
            current_edges -= 1
        
        # Now greedily add back
        current_edges = greedy_add(adj, current_edges)
        return current_edges
    
    best_edges = 0
    best_adj = [0] * n
    
    while time.time() - start_time < timeout - 5:
        # 1. Initialize with varying density
        p = random.uniform(0.40, 0.80)
        adj = [0] * n
        for i, j in all_pairs:
            if random.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        
        # 2. If invalid, use targeted removal to fix
        if conflict_count(adj) > 0:
            current_edges = count_edges(adj)
            current_edges = targeted_remove(adj, current_edges, max_rounds=100)
        else:
            current_edges = count_edges(adj)
        
        # 3. Greedy add
        current_edges = greedy_add(adj, current_edges)
        
        # 4. Simulated annealing for local search
        sa_edges, sa_adj, improved = simulated_annealing(adj, current_edges, max_iter=3000)
        if improved or sa_edges > current_edges:
            adj = sa_adj
            current_edges = sa_edges
        
        # 5. Two-edge swaps for structure improvement
        for _ in range(2):
            if time.time() - start_time > timeout - 5:
                break
            if two_edge_swap(adj, current_edges, max_attempts=200):
                # After successful swap, try greedy add
                current_edges = greedy_add(adj, current_edges)
        
        # 6. Final greedy add
        current_edges = greedy_add(adj, current_edges)
        
        # Update best
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [a for a in adj]
        
        if time.time() - start_time > timeout - 8:
            break
    
    edges = []
    for i, j in all_pairs:
        if (best_adj[i] >> j) & 1:
            edges.append([i, j])
    
    return edges
