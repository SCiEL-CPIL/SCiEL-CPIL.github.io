"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def max_sum_free(elements, max_time=30.0):
        n = len(elements)
        if n <= 3:
            return list(elements)
        
        # 1. Greedy initialization: sort by absolute value descending
        #    Large numbers are less likely to be sums of two others in the set
        #    (since sums tend to be larger than summands, but here we have mixed signs).
        #    Actually, for mixed signs, it's complex. 
        #    Let's use a heuristic: elements with high absolute value are "safer" 
        #    because it's harder to form them as a sum of two smaller magnitude numbers 
        #    if the set is sparse, but actually the opposite is true for sums.
        #    Wait: a+b=c. If c is large, a and b must be large. 
        #    If we pick large numbers, their sums are even larger (or negative large).
        #    If the range is bounded, large numbers have fewer partners that sum to another in-set number.
        
        sorted_elems = sorted(elements, key=lambda x: abs(x), reverse=True)
        
        S = []
        S_set = set()
        
        # Greedy add
        for x in sorted_elems:
            # Check if adding x violates sum-free property
            # x = a + b  (x is sum) -> need a, b in S
            # a + b = x  (x is summand) -> need a in S such that x-a in S, OR b in S such that x-b in S
            # Since a=b allowed, we check:
            # 1. Is x a sum of two elements in S?
            #    For a in S: if (x - a) in S: conflict
            # 2. Is x part of a sum with another element in S?
            #    For a in S: if (x + a) in S: conflict (x + a = c, c in S)
            #    Note: if x+a = x, then a=0, but 0 is not in set.
            
            conflict = False
            for a in S:
                # Case 1: a + b = x => b = x - a
                if (x - a) in S_set:
                    conflict = True
                    break
                # Case 2: a + x = c => c = a + x
                # If c is in S, conflict.
                if (a + x) in S_set:
                    conflict = True
                    break
                # Case 3: x + x = c => c = 2x
                if (2 * x) in S_set:
                    conflict = True
                    break
            
            if not conflict:
                S.append(x)
                S_set.add(x)
        
        # 2. Local Search / Improvement
        # Try to add elements we skipped, or swap elements to allow adding more.
        # Simpler: Try to add any remaining element if it fits.
        remaining = [x for x in elements if x not in S_set]
        
        # Shuffle remaining for variety in local search
        rng_local = Random(42)
        rng_local.shuffle(remaining)
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for x in remaining:
                if time.time() > max_time:
                    break
                # Check if x can be added to current S
                conflict = False
                for a in S:
                    if (x - a) in S_set or (a + x) in S_set:
                        conflict = True
                        break
                if (2 * x) in S_set:
                    conflict = True
                    break
                
                if not conflict:
                    S.append(x)
                    S_set.add(x)
                    remaining.remove(x)
                    improved = True
        
        # 3. Advanced Local Search: Swap
        # Try swapping an element in S with an element not in S to potentially allow more additions later?
        # This is complex. Given time constraints, the greedy + fill is likely strong.
        # Let's add a simple "try to remove one element to add two" heuristic if time permits.
        
        if time.time() < max_time - 2.0:
            # Try to improve by removing 1 and adding 2
            # This is O(N^3) roughly, might be slow.
            # Let's stick to the greedy fill which is O(N^2).
            pass

        return S

    def spectral_max_cut(elements, max_time=30.0):
        """
        Model sum-free constraint as a graph problem.
        Vertices: elements.
        Edges: (u, v) if u+v is in set, or 2u is in set (self-loop handled separately or as edge to self? No, 2u=c means u is adjacent to... wait.
        The constraint is: no a, b, c in S such that a+b=c.
        This is not a standard 2-coloring (Max Cut) problem directly because the "edge" depends on a third vertex c.
        However, we can define a hypergraph or a graph where:
        If we select a subset S, it is invalid if there exists a, b, c in S with a+b=c.
        
        Alternative interpretation for Max Cut:
        Let's define a graph G where vertices are the elements.
        We want to partition V into S (sum-free) and V\S.
        This is not a Max Cut problem on a simple graph because the constraint is 3-uniform (a+b=c).
        
        However, we can approximate it.
        Let's define edges between x and y if x+y is in the set of ALL elements.
        If we choose S, we want to avoid edges (x, y) where x, y in S AND x+y in S.
        This is a "triangle-free" like constraint but directed/ordered.
        
        Actually, let's look at the structure:
        If we fix the set of "sums" C and "summands" A, B...
        
        Let's try a different spectral approach:
        Construct a graph where x and y are connected if x+y is in the set.
        If we pick an independent set in this graph, it is NOT necessarily sum-free.
        Example: {1, 2, 3}. Edges: (1,2) because 1+2=3 in set. (1,3) because 1+3=4 not in set. (2,3) because 2+3=5 not in set.
        Independent set {1, 3} is sum-free? 1+3=4 (not in set). 1+1=2 (not in {1,3}). 3+3=6 (not in set). Yes.
        But {1, 2} is NOT independent (edge exists), and 1+2=3 (not in {1,2}), so {1,2} IS sum-free.
        So independent set in this graph is a SUFFICIENT condition for sum-free, but not necessary.
        Wait, if x+y is in the set, and we pick x and y, we MUST NOT pick x+y.
        So if we pick x and y, we are forced to exclude x+y.
        
        Let's try a Max-Cut on a graph defined as follows:
        Vertices: elements.
        Edge (x, y) if x+y is in the set.
        If we color x and y differently, we satisfy the constraint "if x,y in S then x+y not in S"?
        No. Max Cut partitions into A and B.
        If we say S = A, then for any edge (x, y) with x, y in A, x+y is in the set.
        This doesn't directly prevent x+y from being in A.
        
        Let's try a different graph:
        Vertices: elements.
        Edge (x, x+y) if x, y, x+y are all in the set? No, that's 3-uniform.
        
        Let's go back to the directive: "Model the sum-free constraint as a 2-coloring problem on the Schur graph... and use spectral bisection or max-cut heuristics".
        
        The Schur graph usually has edges (x, y) if x+y is in the set? Or (x, y, z) if x+y=z?
        If we treat it as a graph where x and y are connected if x+y is in the set, and we seek a Max Cut, we are trying to separate "summands" from "sums"?
        
        Actually, a known heuristic for sum-free sets is to take all positive numbers > n/2.
        Another is to take all odd numbers.
        
        Let's try a Spectral Method based on the adjacency matrix of the "sum graph".
        Let G be a graph where x ~ y if x+y is in the set.
        The Laplacian of this graph encodes the structure.
        The Fiedler vector (second smallest eigenvector) gives a partition.
        If we take the positive side of the Fiedler vector as S, does it help?
        
        Let's implement this.
        """
        n = len(elements)
        if n < 5:
            return max_sum_free(elements, max_time)
        
        # Build adjacency matrix
        # A[i][j] = 1 if elements[i] + elements[j] is in elements
        # Note: This graph is undirected if we consider x+y=y+x.
        # But the sum x+y must be in the set.
        
        elem_set = set(elements)
        # Map value to index for quick lookup
        val_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Initialize adjacency matrix (symmetric)
        # Using sparse or dense? n is max 240. Dense 240x240 is fine.
        A = [[0]*n for _ in range(n)]
        
        for i in range(n):
            xi = elements[i]
            for j in range(i, n): # Symmetric
                xj = elements[j]
                s = xi + xj
                if s in elem_set:
                    A[i][j] = 1
                    if i != j:
                        A[j][i] = 1
        
        # Compute Laplacian L = D - A
        # D[i] = sum(A[i])
        D = [sum(A[i]) for i in range(n)]
        
        # We want to solve L v = lambda v
        # Since scipy is available, we can use scipy.sparse.linalg.eigsh
        # But for small n, numpy.linalg.eigh is faster and sufficient.
        
        import numpy as np
        
        L = np.array(D, dtype=float).reshape(n, 1) - np.array(A, dtype=float)
        # L = diag(D) - A
        L = np.diag(D) - np.array(A, dtype=float)
        
        try:
            # Get eigenvalues and eigenvectors
            # We want the second smallest eigenvalue (Fiedler vector)
            # For disconnected graphs, lambda_1 = 0, lambda_2 = 0.
            # We want the eigenvector corresponding to the smallest positive eigenvalue, or just the 2nd smallest if connected.
            # Actually, for Max Cut, we often use the eigenvector of the smallest non-zero eigenvalue.
            
            eigvals, eigvecs = np.linalg.eigh(L)
            
            # Sort by eigenvalue
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Find the first non-zero eigenvalue (or close to it)
            # If the graph is connected, eigvals[0] is 0, eigvals[1] is the Fiedler value.
            # If disconnected, multiple zeros.
            # Let's pick the eigenvector corresponding to eigvals[1] if eigvals[0] is close to 0.
            # If eigvals[1] is also close to 0, the graph is disconnected, and the eigenvector might be constant on components.
            # In that case, we might need to look further.
            
            # Heuristic: Pick the eigenvector that gives the best cut quality?
            # Or just take the 2nd eigenvector.
            
            # Let's try the eigenvector corresponding to the smallest positive eigenvalue.
            # If all are 0 (empty graph), return all elements? No, that's not sum-free.
            # If the graph is empty, no edges, any subset is valid? No, the graph edges represent constraints.
            # If A is all 0s, then no x+y is in the set. Then ANY subset is sum-free?
            # Wait. If A[i][j] = 1 iff x_i + x_j in set.
            # If A is all 0s, then for any x, y in set, x+y is NOT in set.
            # This implies the set itself is sum-free?
            # Let's check: Sum-free means no a, b, c in S with a+b=c.
            # If we take S = elements, and for all a, b in S, a+b is not in S, then S is sum-free.
            # So if A is all 0s, we can return all elements.
            
            if np.allclose(A, 0):
                return list(elements)
                
            # Find the index of the smallest positive eigenvalue
            # Tolerance for zero
            tol = 1e-8
            fiedler_idx = 1
            if abs(eigvals[0]) < tol:
                fiedler_idx = 1
                if abs(eigvals[1]) < tol:
                    # Disconnected graph
                    # Find the first non-zero
                    for k in range(len(eigvals)):
                        if abs(eigvals[k]) > tol:
                            fiedler_idx = k
                            break
                    else:
                        # All zero? Should not happen if A is not all 0s.
                        return list(elements)
            
            v = eigvecs[:, fiedler_idx]
            
            # Partition based on sign of v
            # v[i] > 0 -> in S, v[i] < 0 -> not in S
            # Threshold at 0
            
            S_indices = [i for i in range(n) if v[i] > 0]
            S = [elements[i] for i in S_indices]
            
            # This S is a candidate. But is it sum-free?
            # The spectral partition minimizes the cut of the graph A.
            # Edges crossing the cut are (x, y) where x in S, y not in S, and x+y in set.
            # Edges NOT crossing (both in S) are (x, y) where x, y in S and x+y in set.
            # We want to MINIMIZE edges within S.
            # Max Cut MAXIMIZES edges crossing.
            # So Min Cut of A (or Max Cut) should minimize internal edges.
            # If x, y in S and x+y in set, that's an internal edge.
            # So this heuristic tries to minimize the number of pairs (x, y) in S such that x+y is in the original set.
            # This is a necessary condition for sum-free, but not sufficient (x+y could be in S).
            # However, it's a good structural hint.
            
            # Now, refine S using the local search from max_sum_free
            # We can use S as the initial set for the greedy/local search, or use it to prune.
            
            # Let's initialize the local search with this S.
            # But max_sum_free starts from scratch.
            # Let's create a modified local search that starts from S.
            
            S_set = set(S)
            remaining = [x for x in elements if x not in S_set]
            
            # Check if S is sum-free. If not, we need to remove elements.
            # The spectral method doesn't guarantee sum-free.
            # So we must run a repair step.
            
            # Repair: Remove elements that cause conflicts.
            # Or, simpler: Use the spectral result to ORDER the elements for the greedy step.
            # Elements with high v[i] are "more likely" to be in the sum-free set.
            
            # Let's sort elements by v[i] descending and run the greedy algorithm.
            # This leverages the global structure to guide the local greedy.
            
            sorted_by_spectral = sorted(elements, key=lambda x: v[val_to_idx[x]], reverse=True)
            
            S_final = []
            S_final_set = set()
            
            for x in sorted_by_spectral:
                conflict = False
                for a in S_final:
                    if (x - a) in S_final_set or (a + x) in S_final_set:
                        conflict = True
                        break
                if (2 * x) in S_final_set:
                    conflict = True
                    break
                
                if not conflict:
                    S_final.append(x)
                    S_final_set.add(x)
            
            # Now run local search improvement on S_final
            remaining = [x for x in elements if x not in S_final_set]
            rng_local = Random(42)
            rng_local.shuffle(remaining)
            
            improved = True
            while improved and time.time() < max_time:
                improved = False
                for x in remaining:
                    if time.time() > max_time:
                        break
                    conflict = False
                    for a in S_final:
                        if (x - a) in S_final_set or (a + x) in S_final_set:
                            conflict = True
                            break
                    if (2 * x) in S_final_set:
                        conflict = True
                        break
                    
                    if not conflict:
                        S_final.append(x)
                        S_final_set.add(x)
                        remaining.remove(x)
                        improved = True
            
            return S_final
            
        except Exception as e:
            # Fallback to standard greedy if spectral fails
            return max_sum_free(elements, max_time)

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        if time.time() > deadline:
            results.append([])
            continue
            
        elems = instance(size, seed)
        
        time_budget = (size / 420.0) * 100.0
        
        remaining_time = deadline - time.time()
        if time_budget > remaining_time:
            time_budget = max(1.0, remaining_time - 1.0)
            
        # Try spectral method first, fallback to greedy
        # Spectral method might be slower due to eigen decomposition, but n is small.
        # For n=240, eigh is very fast (<0.1s).
        
        subset = spectral_max_cut(elems, max_time=time_budget)
        
        # Verify sum-free property just in case
        if not is_sum_free(subset):
             # Fallback if spectral result is invalid (shouldn't happen due to repair, but safe)
             subset = max_sum_free(elems, max_time=time_budget)
             
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
