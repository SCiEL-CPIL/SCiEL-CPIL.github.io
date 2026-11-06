"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
import time
import random

def construct():
    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def check_sum_free(subset):
        """Verify subset is sum-free."""
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def spectral_max_cut(elements):
        """
        Model sum-free constraint as 2-coloring on Schur graph.
        Vertices: elements. Edge between x and y if:
          - 2x = y (so x+x=y, can't have both x and y)
          - x+y = z for some z in set (so x, y, z form a Schur triple)
        
        We build a graph where edges represent "cannot both be in sum-free set"
        in a simplified way: if x+y=z, we add edges (x,z) and (y,z) to force
        that at least one of each pair must be excluded.
        
        Use spectral bisection (Fiedler vector) to partition, then refine.
        """
        n = len(elements)
        if n == 0:
            return []
        
        elem_set = set(elements)
        elem_to_idx = {v: i for i, v in enumerate(elements)}
        
        # Build Schur graph: edge (i,j) if elements[i] + elements[j] is in elem_set
        # This captures: if we have both elements[i] and elements[j], we can't have elements[i]+elements[j]
        # We model this as: add edge between i and k where k = idx of elements[i]+elements[j]
        # and between j and k. This way, a proper 2-coloring ensures no Schur triple.
        
        row = []
        col = []
        
        for i in range(n):
            for j in range(i, n):  # i <= j to handle a=a case
                s = elements[i] + elements[j]
                if s in elem_to_idx:
                    k = elem_to_idx[s]
                    # Edge between i and k
                    if i != k:
                        row.append(i)
                        col.append(k)
                    # Edge between j and k
                    if j != k:
                        row.append(j)
                        col.append(k)
        
        if not row:
            # No conflicts, return all
            return list(elements)
        
        # Build sparse adjacency matrix
        A = lil_matrix((n, n), dtype=np.float64)
        for r, c in zip(row, col):
            A[r, c] = 1
            A[c, r] = 1
        A = A.tocsr()
        
        # Compute Fiedler vector (eigenvector of second-smallest eigenvalue of Laplacian)
        D = np.array(A.sum(axis=1)).flatten()
        L = csr_matrix((n, n), dtype=np.float64)
        # L = D - A
        L = (csr_matrix(np.diag(D)) - A).tocsr()
        
        # Use scipy to find eigenvalues/eigenvectors
        from scipy.sparse.linalg import eigsh
        try:
            # Find the smallest eigenvalue (0) and the second smallest (Fiedler)
            # eigsh finds eigenvalues near sigma
            vals, vecs = eigsh(L, k=2, sigma=0.0, which='LM')
            # The Fiedler vector is the eigenvector with the second smallest eigenvalue
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
        except Exception:
            # Fallback: use random vector
            fiedler = np.random.randn(n)
        
        # Partition: positive -> group 0, negative -> group 1
        partition = (fiedler > 0).astype(int)
        
        # Get the two partitions
        group0 = [elements[i] for i in range(n) if partition[i] == 0]
        group1 = [elements[i] for i in range(n) if partition[i] == 1]
        
        # For each group, we need to check if it's sum-free.
        # If not, we need to remove elements to make it sum-free.
        # Take the larger group and clean it up.
        
        def clean_sum_free(group):
            """Remove elements from group to make it sum-free."""
            if not group:
                return []
            current = set(group)
            # Iteratively remove elements that are part of Schur triples
            changed = True
            while changed:
                changed = False
                elems_list = list(current)
                for i in range(len(elems_list)):
                    for j in range(i, len(elems_list)):
                        s = elems_list[i] + elems_list[j]
                        if s in current:
                            # Remove s (the sum) - heuristic: remove the "result"
                            current.discard(s)
                            changed = True
                            break
                    if changed:
                        break
            return list(current)
        
        clean0 = clean_sum_free(group0)
        clean1 = clean_sum_free(group1)
        
        # Try both clean groups and pick the larger
        best = clean0 if len(clean0) >= len(clean1) else clean1
        
        # Also try the parity-based approach as a baseline
        odd_subset = [x for x in elements if x % 2 != 0]
        if len(odd_subset) > len(best):
            best = odd_subset
        
        # Try to add elements from the other group to best
        remaining = [x for x in elements if x not in set(best)]
        for x in remaining:
            can_add = True
            for a in best:
                if a + x in set(best):
                    can_add = False
                    break
                if 2 * a == x:
                    can_add = False
                    break
            if can_add:
                # Also check pairs in best that sum to x
                for i in range(len(best)):
                    for j in range(i, len(best)):
                        if best[i] + best[j] == x:
                            can_add = False
                            break
                    if not can_add:
                        break
            if can_add:
                best.append(x)
        
        return best

    def local_search_improve(elements, initial_subset):
        """Improve a sum-free subset via local search (add/remove swaps)."""
        current = list(initial_subset)
        current_set = set(current)
        remaining = [x for x in elements if x not in current_set]
        
        def is_valid(candidate, s):
            """Check if candidate can be added to set s without violating sum-free."""
            for a in s:
                if a + candidate in s:
                    return False
                if 2 * a == candidate:
                    return False
            for a in s:
                for b in s:
                    if a + b == candidate:
                        return False
            return True
        
        improved = True
        while improved:
            improved = False
            for x in remaining[:]:
                if is_valid(x, current_set):
                    current.append(x)
                    current_set.add(x)
                    remaining.remove(x)
                    improved = True
        
        # Try swaps: remove one element, try to add two
        improved = True
        while improved:
            improved = False
            for i in range(len(current)):
                for x in remaining:
                    # Try removing current[i] and adding x
                    temp_set = current_set - {current[i]}
                    if is_valid(x, temp_set):
                        # Also check if we can add another element
                        current[i] = x
                        current_set.discard(current[i])
                        current_set.add(x)
                        remaining.remove(x)
                        improved = True
                        break
                if improved:
                    break
        
        return current

    def sum_free_spectral(elements):
        """Main solver: spectral bisection + local search refinement."""
        n = len(elements)
        if n == 0:
            return []
        
        best_subset = []
        best_size = 0
        
        # Strategy 1: Spectral bisection
        spec_result = spectral_max_cut(elements)
        spec_clean = local_search_improve(elements, spec_result)
        if len(spec_clean) > best_size:
            best_size = len(spec_clean)
            best_subset = spec_clean
        
        # Strategy 2: Parity-based + local search
        odd_subset = [x for x in elements if x % 2 != 0]
        odd_clean = local_search_improve(elements, odd_subset)
        if len(odd_clean) > best_size:
            best_size = len(odd_clean)
            best_subset = odd_clean
        
        # Strategy 3: Multiple random restarts with spectral init
        rng_local = random.Random(42)
        for restart in range(10):
            # Perturb the spectral partition
            spec_result2 = spectral_max_cut(elements)
            # Randomly flip some elements
            perturbed = []
            for x in spec_result2:
                if rng_local.random() < 0.3:
                    continue  # skip some
                perturbed.append(x)
            # Fill with random elements
            if len(perturbed) < n // 2:
                extras = rng_local.sample(elements, min(n // 2 - len(perturbed), n))
                for x in extras:
                    if x not in perturbed:
                        perturbed.append(x)
            
            # Clean and improve
            cleaned = local_search_improve(elements, perturbed)
            if len(cleaned) > best_size:
                best_size = len(cleaned)
                best_subset = cleaned
        
        # Final verification and cleanup
        if not check_sum_free(best_subset):
            # Fallback to parity
            best_subset = [x for x in elements if x % 2 != 0]
        
        return best_subset

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        subset = sum_free_spectral(elements)
        results.append(subset)
    
    return results
# EVOLVE-BLOCK-END
