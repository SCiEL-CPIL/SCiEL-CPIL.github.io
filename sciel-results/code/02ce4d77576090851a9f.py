import time
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    total = q ** n
    all_centers = np.array(list(product(range(q), repeat=n)), dtype=np.uint8)
    
    # For each center, compute the flat indices of points it covers
    # Center c covers: c itself, and for each position i, c with position i changed to each other value
    def compute_coverage(c):
        """Return array of flat point indices covered by center c"""
        pts = [int(np.ravel_multi_index(c, (q,)*n))]
        for i in range(n):
            for v in range(q):
                if v != c[i]:
                    p = list(c)
                    p[i] = v
                    pts.append(int(np.ravel_multi_index(p, (q,)*n)))
        return np.array(pts, dtype=np.int32)
    
    # Precompute all coverages
    print("Precomputing coverages...")
    coverages = [compute_coverage(c) for c in all_centers]
    print(f"Done. Time: {time.time()-start_time:.1f}s")
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    selected = []
    remaining = set(range(len(coverages)))
    
    # For efficiency, track for each point which centers cover it
    # point_to_centers[p] = list of center indices that cover point p
    point_to_centers = [[] for _ in range(total)]
    for ci, cov in enumerate(coverages):
        for p in cov:
            point_to_centers[p].append(ci)
    
    print(f"Building point-to-center mapping done. Time: {time.time()-start_time:.1f}s")
    
    while uncovered.any() and remaining:
        if time.time() > deadline:
            break
        
        # Find center covering most uncovered points
        best_ci = -1
        best_count = 0
        
        for ci in remaining:
            cov = coverages[ci]
            count = np.sum(uncovered[cov])
            if count > best_count:
                best_count = count
                best_ci = ci
        
        if best_ci == -1:
            break
        
        selected.append(all_centers[best_ci])
        uncovered[coverages[best_ci]] = False
        remaining.discard(best_ci)
        
        if len(selected) % 10 == 0:
            print(f"  {len(selected)} selected, {int(uncovered.sum())} uncovered, time={time.time()-start_time:.1f}s")
    
    # Iterative removal
    cover_count = np.zeros(total, dtype=np.int32)
    center_indices = []
    for c in selected:
        ci = int(np.where((all_centers == c).all(axis=1))[0][0])
        center_indices.append(ci)
        cover_count[coverages[ci]] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(center_indices)):
            if time.time() > deadline:
                break
            ci = center_indices[i]
            cov = coverages[ci]
            if np.all(cover_count[cov] > 1):
                cover_count[cov] -= 1
                center_indices.pop(i)
                changed = True
                break
    
    selected = [all_centers[ci] for ci in center_indices]
    print(f"Final: {len(selected)} codewords, time={time.time()-start_time:.1f}s")
    
    return [list(c) for c in selected]
