"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    d = 7
    n = 2 ** d
    
    # Precompute all edges
    all_edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(all_edges)  # 448 edges
    
    # Precompute for each edge, the triples of other edges that form a C4 with it
    # Edge (x, y) where y = x ^ (1<<i)
    # For each j != i, the face with directions i,j has vertices x, y, x^(1<<j), y^(1<<j)
    # The 4 edges are: (x,y) dir i, (x^(1<<j), y^(1<<j)) dir i (parallel to (x,y))
    #                   (x, x^(1<<j)) dir j, (y, y^(1<<j)) dir j (parallel to each other)
    # A C4 is formed if all 4 edges are selected. So if edge idx is selected,
    # we cannot have all three of the other edges selected.
    
    forbidden_triples = [[] for _ in range(m)]
    for idx, (x, y) in enumerate(all_edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            e1 = (min(x, xj), max(x, xj))
            e2 = (min(y, yj), max(y, yj))
            e3 = (min(xj, yj), max(xj, yj))
            idx1 = edge_to_idx.get(e1)
            idx2 = edge_to_idx.get(e2)
            idx3 = edge_to_idx.get(e3)
            if idx1 is not None and idx2 is not None and idx3 is not None:
                forbidden_triples[idx].append((idx1, idx2, idx3))
    
    def can_add(idx, chosen):
        """Check if edge idx can be added without creating a C4."""
        for (i1, i2, i3) in forbidden_triples[idx]:
            if chosen[i1] and chosen[i2] and chosen[i3]:
                return False
        return True
    
    def greedy_polish(chosen, count):
        """Try to add all possible edges in multiple passes."""
        improved = True
        while improved:
            improved = False
            for idx in range(m):
                if not chosen[idx] and can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    improved = True
        return count
    
    # Local search with simulated annealing
    def local_search(chosen, count, T_init=3.0, T_min=0.01, cooling=0.99998, max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        best_count = count
        best_chosen = chosen.copy()
        no_improve = 0
        max_no_improve = 30000
        
        T = T_init
        while time.time() < max_time and T > T_min:
            idx = random.randint(0, m - 1)
            
            if not chosen[idx]:
                if can_add(idx, chosen):
                    chosen[idx] = True
                    count += 1
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Try 2-opt swap: remove a conflicting edge, add idx
                    conflicts = []
                    for (i1, i2, i3) in forbidden_triples[idx]:
                        if chosen[i1] and chosen[i2] and chosen[i3]:
                            conflicts.extend([i1, i2, i3])
                    
                    if conflicts:
                        # Try a few random conflicting edges
                        random.shuffle(conflicts)
                        success = False
                        for remove_idx in conflicts[:3]:
                            if not chosen[remove_idx]:
                                continue
                            # Remove remove_idx
                            chosen[remove_idx] = False
                            count -= 1
                            if can_add(idx, chosen):
                                chosen[idx] = True
                                count += 1
                                success = True
                                if count > best_count:
                                    best_count = count
                                    best_chosen = chosen.copy()
                                    no_improve = 0
                                else:
                                    no_improve += 1
                                break
                            else:
                                # Revert
                                chosen[remove_idx] = True
                                count += 1
                        if not success:
                            no_improve += 1
                    else:
                        no_improve += 1
            else:
                # Consider removing idx
                # Only remove if it enables adding 2+ edges
                chosen[idx] = False
                count -= 1
                added = 0
                added_edges = []
                for other in range(m):
                    if not chosen[other] and can_add(other, chosen):
                        chosen[other] = True
                        count += 1
                        added += 1
                        added_edges.append(other)
                
                if added > 1:
                    # Net gain, keep
                    if count > best_count:
                        best_count = count
                        best_chosen = chosen.copy()
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Revert
                    for other in added_edges:
                        chosen[other] = False
                        count -= 1
                    chosen[idx] = True
                    count += 1
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                chosen = best_chosen.copy()
                count = best_count
                T = max(T, 1.5)
                no_improve = 0
        
        return best_chosen, best_count
    
    # Multi-round search with different initializations
    best_count = 0
    best_chosen = None
    
    # Round 1: Spectral initialization
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    
    # Build conflict graph adjacency
    row = []
    col = []
    data = []
    for i in range(m):
        for (i1, i2, i3) in forbidden_triples[i]:
            # i conflicts with i1, i2, i3 in the sense that they can't all be selected
            # But for spectral init, we build a graph where edges represent "can't be together"
            # Actually, let's use a different approach: build a graph where we want to maximize
            # the number of edges selected such that no forbidden triple is fully selected.
            # This is a 3-uniform hypergraph independent set problem.
            # For spectral init, let's just use random initialization with different densities.
            pass
    
    # Use random initialization with greedy polish
    random.seed(42)
    for round_num in range(5):
        if time.time() > deadline - 10:
            break
        
        if round_num == 0:
            # Start with all edges, then remove to fix conflicts
            chosen = np.ones(m, dtype=bool)
            count = m
            # Remove edges to fix all C4s
            # For each C4 (forbidden triple + the edge), if all 4 are selected, remove one
            # This is a hitting set / vertex cover on hypergraph problem
            # Greedy: repeatedly find a violated C4 and remove a random edge from it
            violated = []
            for idx in range(m):
                for (i1, i2, i3) in forbidden_triples[idx]:
                    if chosen[idx] and chosen[i1] and chosen[i2] and chosen[i3]:
                        violated.append((idx, i1, i2, i3))
            
            random.shuffle(violated)
            for (a, b, c, e) in violated:
                if chosen[a] and chosen[b] and chosen[c] and chosen[e]:
                    # Remove a random one
                    to_remove = random.choice([a, b, c, e])
                    chosen[to_remove] = False
                    count -= 1
            
            count = greedy_polish(chosen, count)
        else:
            # Random initialization with varying density
            density = 0.3 + 0.1 * round_num
            random.seed(42 + round_num * 100)
            chosen = np.zeros(m, dtype=bool)
            for idx in range(m):
                if random.random() < density:
                    if can_add(idx, chosen):
                        chosen[idx] = True
            count = chosen.sum()
            count = greedy_polish(chosen, count)
        
        if best_count == 0 or count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        
        # Local search
        remaining = deadline - time.time() - 5
        if remaining > 5:
            chosen, count = local_search(chosen, count, T_init=2.0, T_min=0.01, cooling=0.99998, max_time=remaining)
            count = greedy_polish(chosen, count)
            if count > best_count:
                best_count = count
                best_chosen = chosen.copy()
    
    # Final polish
    count = greedy_polish(best_chosen, best_count)
    if count > best_count:
        best_count = count
    
    # Convert to output
    out = []
    for idx in range(m):
        if best_chosen[idx]:
            x, y = all_edges[idx]
            out.append([x, y])
    
    return out
# EVOLVE-BLOCK-END
