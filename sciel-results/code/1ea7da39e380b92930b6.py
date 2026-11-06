import time
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import random

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    aps = []
    pos_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            ap_idx = len(aps)
            aps.append((p1, p2, p3, p4))
            pos_aps[p1].append(ap_idx)
            pos_aps[p2].append(ap_idx)
            pos_aps[p3].append(ap_idx)
            pos_aps[p4].append(ap_idx)
    
    total_aps = len(aps)
    
    def count_colors(colors):
        return len(set(colors))
    
    def is_valid(colors):
        """Check if every 4-AP has at least 3 distinct colors."""
        for ap in aps:
            p1, p2, p3, p4 = ap
            if len({colors[p1], colors[p2], colors[p3], colors[p4]}) < 3:
                return False
        return True
    
    def spectral_clustering(k):
        """
        Use spectral clustering on the 4-AP incidence structure to find
        k natural color classes.
        """
        # Build incidence matrix: positions x APs
        # A[i, j] = 1 if position i is in AP j
        rows = []
        cols = []
        for ap_idx, (p1, p2, p3, p4) in enumerate(aps):
            for p in (p1, p2, p3, p4):
                rows.append(p)
                cols.append(ap_idx)
        
        A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, total_aps))
        
        # Build Laplacian: L = D - A @ A.T
        # where D is diagonal of row sums of A @ A.T
        AAT = A @ A.T
        D = csr_matrix((np.array(AAT.sum(axis=1)).flatten(), (range(N), range(N))), shape=(N, N))
        L = (D - AAT).toarray()
        
        # Compute smallest k eigenvectors of L
        try:
            # Use eigsh for smallest eigenvalues
            vals, vecs = eigsh(L, k=k, which='SM')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vecs = vecs[:, idx]
            
            # Normalize rows
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
            
            # K-means on the normalized eigenvectors
            from scipy.cluster.vq import kmeans2
            centroids, labels = kmeans2(vecs, k, minit='++')
            
            return labels.tolist()
        except Exception:
            # Fallback: use position-based coloring
            return [i % k for i in range(N)]
    
    def greedy_init(k):
        """Greedy coloring that tries to use k colors."""
        colors = [-1] * N
        for i in range(N):
            # Try each color in order
            for c in range(k):
                colors[i] = c
                valid = True
                for ap_idx in pos_aps[i]:
                    p1, p2, p3, p4 = aps[ap_idx]
                    assigned = [colors[p] for p in (p1, p2, p3, p4) if colors[p] != -1]
                    unassigned_count = sum(1 for p in (p1, p2, p3, p4) if colors[p] == -1)
                    distinct = set(assigned)
                    if len(distinct) + unassigned_count < 3:
                        valid = False
                        break
                    if unassigned_count == 0 and len(distinct) < 3:
                        valid = False
                        break
                if valid:
                    break
            else:
                # No color works, use first available
                colors[i] = 0
        return colors
    
    def merge_colors(colors, k):
        """
        Try to merge color classes to reduce color count while maintaining validity.
        Returns new coloring with fewer colors if possible.
        """
        colors = list(colors)
        current_k = count_colors(colors)
        
        # Map actual color values to 0..current_k-1
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        colors = [color_map[c] for c in colors]
        
        # Try to merge pairs of colors
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_k = count_colors(colors)
            if current_k <= 3:
                break
            
            # Get list of colors present
            present_colors = sorted(set(colors))
            
            # Try all pairs
            for i in range(len(present_colors)):
                for j in range(i + 1, len(present_colors)):
                    ci = present_colors[i]
                    cj = present_colors[j]
                    
                    # Tentatively merge cj into ci
                    test_colors = [ci if c == cj else c for c in colors]
                    
                    # Check validity
                    if is_valid(test_colors):
                        colors = test_colors
                        improved = True
                        break
                if improved:
                    break
        
        # Renumber to 0..k-1
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        colors = [color_map[c] for c in colors]
        
        return colors
    
    def local_search(colors, k, max_iter=10000):
        """
        Local search to minimize color count by reassigning individual positions.
        """
        colors = list(colors)
        current_k = count_colors(colors)
        
        # Renumber
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        colors = [color_map[c] for c in colors]
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            if current_k <= 3:
                break
            
            # Try to reassign a random position to a color that reduces total colors
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            
            # Find colors that are present
            present_colors = sorted(set(colors))
            
            # Try each present color (except old_color)
            for new_color in present_colors:
                if new_color == old_color:
                    continue
                
                # Check if new_color would be the last of its kind
                # (i.e., removing old_color's last instance and adding to new_color)
                old_count = sum(1 for c in colors if c == old_color)
                new_count = sum(1 for c in colors if c == new_color)
                
                # Only try if this might reduce color count
                if old_count > 1:
                    continue
                
                # Tentatively assign
                colors[pos] = new_color
                
                # Check validity
                if is_valid(colors):
                    # Check if color count decreased
                    new_k = count_colors(colors)
                    if new_k < current_k:
                        current_k = new_k
                        improved = True
                        break
                    else:
                        colors[pos] = old_color
            else:
                # No improvement, try random reassignment
                if random.random() < 0.3:
                    new_color = random.choice(present_colors)
                    colors[pos] = new_color
                    if is_valid(colors):
                        current_k = count_colors(colors)
                    else:
                        colors[pos] = old_color
        
        # Final renumber
        unique_colors = sorted(set(colors))
        color_map = {c: i for i, c in enumerate(unique_colors)}
        colors = [color_map[c] for c in colors]
        
        return colors
    
    # Strategy:
    # 1. Start with spectral clustering for k=4,5,6
    # 2. Merge colors iteratively
    # 3. Local search to minimize
    
    best_colors = None
    best_k = N  # Start with worst case
    
    # Try different initial k values
    for init_k in [4, 5, 6, 8]:
        if time.time() > deadline - 10:
            break
        
        # Get initial coloring from spectral clustering
        colors = spectral_clustering(init_k)
        
        # Ensure it's valid; if not, fix with greedy
        if not is_valid(colors):
            # Fix invalid APs by reassigning
            for _ in range(100):
                if time.time() > deadline - 10:
                    break
                fixed = False
                for ap in aps:
                    p1, p2, p3, p4 = ap
                    distinct = {colors[p1], colors[p2], colors[p3], colors[p4]}
                    if len(distinct) < 3:
                        # Need to change one of these positions
                        # Try changing p4 (last) to a new color
                        for new_c in range(N):
                            colors[p4] = new_c
                            if len({colors[p1], colors[p2], colors[p3], colors[p4]}) >= 3:
                                # Check if this broke other APs
                                if is_valid(colors):
                                    fixed = True
                                    break
                        if fixed:
                            break
                if not fixed:
                    break
        
        if not is_valid(colors):
            continue
        
        # Merge colors
        colors = merge_colors(colors, init_k)
        
        # Local search
        colors = local_search(colors, init_k, max_iter=5000)
        
        k = count_colors(colors)
        if k < best_k:
            best_k = k
            best_colors = colors
        
        if best_k <= 3:
            break
    
    # If we still have a high color count, try greedy + merge
    if best_colors is None or best_k > 4:
        if time.time() > deadline - 5:
            # Return whatever we have or a simple valid coloring
            if best_colors is not None and is_valid(best_colors):
                return best_colors
            # Fallback: use many colors
            colors = list(range(N))
            return colors
        
        # Greedy with k=6
        colors = greedy_init(6)
        if is_valid(colors):
            colors = merge_colors(colors, 6)
            colors = local_search(colors, 6, max_iter=3000)
            k = count_colors(colors)
            if k < best_k:
                best_k = k
                best_colors = colors
    
    # Final check and return
    if best_colors is None:
        # Absolute fallback
        best_colors = list(range(N))
    
    if not is_valid(best_colors):
        # Should not happen, but ensure validity
        # Use a simple valid coloring
        colors = []
        for i in range(N):
            for c in range(N):
                colors.append(c)
                break
            break
        best_colors = list(range(N))
    
    return best_colors
