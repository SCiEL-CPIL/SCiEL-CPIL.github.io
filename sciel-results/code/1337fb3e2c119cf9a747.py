import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj):
        """Enumerate all maximal cliques of size >= 2."""
        n = len(adj)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            pivot_candidates = P | X
            if pivot_candidates:
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = P - adj[pivot]
            else:
                return
                
            for v in list(candidates):
                R.add(v)
                P_new = P & adj[v]
                X_new = X & adj[v]
                bk(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
        bk(set(), set(range(n)), set())
        return cliques

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        cliques = bron_kerbosch(adj)
        
        if not cliques:
            return []
            
        num_cliques = len(cliques)
        
        # Create bitmask for each vertex: bit i is set if vertex is in clique i
        masks = [0] * n
        for i, c in enumerate(cliques):
            bit = 1 << i
            for v in c:
                masks[v] |= bit
                
        all_mask = (1 << num_cliques) - 1
        
        best_solution = None
        best_size = float('inf')
        
        num_restarts = 1000 # We will break based on time
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            rng = random.Random(seed * 1000 + restart)
            
            # 1. Greedy Construction
            # Randomize the order of vertices to break ties and explore different basins
            available = list(range(n))
            rng.shuffle(available)
            
            selected = set()
            covered = 0
            
            while covered != all_mask:
                best_v = -1
                best_gain = -1
                best_score = -1
                
                # To speed up, we can sample if n is large, but n<=40 so full scan is fine.
                # We prioritize vertices that cover more *uncovered* cliques.
                # Tie-breaking: random noise or just random selection among max gain.
                
                candidates = []
                for v in available:
                    if v in selected:
                        continue
                    gain = (masks[v] & ~covered).bit_count()
                    if gain > 0:
                        # Add small random noise to encourage diversity
                        score = gain + rng.random() * 0.01
                        if score > best_score:
                            best_score = score
                            best_gain = gain
                            best_v = v
                            candidates = [v]
                        elif score == best_score:
                            candidates.append(v)
                
                if best_v == -1:
                    # Should not happen if graph is valid, but safety break
                    break
                
                # If multiple candidates had the same max score (due to noise, strictly > is used above, 
                # so we pick the first one found with max score in this iteration? No, we updated best_v on strict >.
                # Let's refine: pick one randomly from the set of vertices with max gain.
                
                # Re-evaluating max gain without noise for candidate list
                max_gain_val = 0
                for v in available:
                    if v in selected: continue
                    g = (masks[v] & ~covered).bit_count()
                    if g > max_gain_val:
                        max_gain_val = g
                        
                if max_gain_val == 0:
                    break
                    
                max_gain_vertices = [v for v in available if (masks[v] & ~covered).bit_count() == max_gain_val and v not in selected]
                v = rng.choice(max_gain_vertices)
                
                selected.add(v)
                covered |= masks[v]
                
            # 2. Local Search / Minimization
            # We have a valid hitting set 'selected'.
            # We want to minimize its size.
            
            sel_list = list(selected)
            
            # Pass 1: Remove redundant vertices
            # Iterate until no improvement.
            
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                # Shuffle order to avoid bias
                rng.shuffle(sel_list)
                
                for v in sel_list:
                    if v not in selected:
                        continue
                    
                    # Check if v is redundant
                    # Calculate coverage without v
                    covered_without_v = 0
                    for u in selected:
                        if u != v:
                            covered_without_v |= masks[u]
                            
                    if covered_without_v == all_mask:
                        selected.discard(v)
                        improved = True
                        # Restart loop to check for further redundancies
                        break # Break inner loop, restart outer while
                    
                if not improved:
                    # Try 1-for-1 swaps
                    # For each v in selected, find cliques only covered by v.
                    # If none, we would have removed it.
                    # So there is at least one critical clique.
                    # We want to replace v with w such that w covers all critical cliques of v,
                    # and the new set is valid.
                    # Ideally, w allows us to remove another vertex later, or w is "better".
                    
                    swap_made = False
                    for v in list(selected):
                        # Find cliques that ONLY v covers in the current set
                        # critical_mask = all_mask & ~(covered_without_v)
                        # But covered_without_v depends on v.
                        
                        covered_without_v = 0
                        for u in selected:
                            if u != v:
                                covered_without_v |= masks[u]
                                
                        critical_mask = all_mask & ~covered_without_v
                        
                        if critical_mask == 0:
                            # v is redundant (should have been caught)
                            selected.discard(v)
                            improved = True
                            swap_made = True
                            break
                        
                        # Find w not in selected such that (masks[w] & critical_mask) == critical_mask
                        # i.e., w covers all critical cliques.
                        candidates = []
                        for w in range(n):
                            if w in selected:
                                continue
                            if (masks[w] & critical_mask) == critical_mask:
                                candidates.append(w)
                                
                        if candidates:
                            # Choose w that maximizes additional coverage? 
                            # Or just pick one randomly.
                            # Picking one that covers more total cliques might help future removals.
                            best_w = max(candidates, key=lambda w: masks[w].bit_count())
                            
                            # Perform swap
                            selected.discard(v)
                            selected.add(best_w)
                            improved = True
                            swap_made = True
                            break # Restart minimization loop to check for new redundancies
                            
                    if swap_made:
                        continue # Restart while loop
                    else:
                        break # No more improvements
                    
        # End while improved
        
        if best_solution is None or len(selected) < best_size:
            best_size = len(selected)
            best_solution = list(selected)
            
        if best_size == 0:
            break
            
    return sorted(best_solution) if best_solution else []

    results = []
    # Solve each instance
    # We share the global deadline.
    # To be safe, we allocate time roughly evenly but allow the last one to use remaining time.
    
    configs = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Estimate time per instance to set individual deadlines?
    # The solve_instance function checks the global `deadline`.
    # We just call it sequentially.
    
    for n, p, seed in configs:
        if time.time() > deadline - 2.0:
            # Emergency fallback: return all vertices in any edge
            adj = random_graph(n, p, seed)
            touched = {x for u in range(n) for v in adj[u] for x in (u, v)}
            results.append(sorted(touched))
        else:
            results.append(solve_instance(n, p, seed))
            
    return results

if __name__ == "__main__":
    # Test locally
    res = construct()
    print(res)
