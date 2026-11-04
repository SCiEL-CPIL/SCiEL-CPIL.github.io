"""Matrix multiplication 4x5x6 rank minimization."""
import time
import sys
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    nm, mp, pn = n * m, m * p, p * n  # 20, 30, 24
    num_points = n * m * p  # 120
    
    # Schoolbook baseline
    best_rank = num_points
    best_U = np.zeros((nm, best_rank))
    best_V = np.zeros((mp, best_rank))
    best_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                pt = i * mp + j * p + k
                best_U[i * m + j, col] = 1
                best_V[j * p + k, col] = 1
                best_W[k * n + i, col] = 1
                col += 1
    
    # Generate box candidates: subsets I⊆[n], J⊆[m], K⊆[p] with small sizes
    # A box (I,J,K) with weight w contributes w to all (i,j,k) with i∈I, j∈J, k∈K
    boxes = []
    for I in range(1, 1 << n):
        if bin(I).count('1') > 2:
            continue
        for J in range(1, 1 << m):
            if bin(J).count('1') > 2:
                continue
            for K in range(1, 1 << p):
                if bin(K).count('1') > 2:
                    continue
                boxes.append((I, J, K))
    
    num_boxes = len(boxes)
    
    # Precompute point lists for each box
    box_points = []
    for I, J, K in boxes:
        pts = []
        for i in range(n):
            if I & (1 << i):
                for j in range(m):
                    if J & (1 << j):
                        for k in range(p):
                            if K & (1 << k):
                                pts.append(i * mp + j * p + k)
        box_points.append(pts)
    
    # Greedy exact cover: repeatedly pick box covering most points with residual >= 1
    residual = np.ones(num_points)
    selected = []
    
    # Sort boxes by size descending for faster greedy
    box_order = sorted(range(num_boxes), key=lambda idx: -len(box_points[idx]))
    
    while np.any(residual > 1e-9) and time.time() < deadline - 10:
        best_box = None
        best_count = 0
        for b_idx in box_order:
            pts = box_points[b_idx]
            count = 0
            for pt in pts:
                if residual[pt] >= 1 - 1e-9:
                    count += 1
            if count > best_count:
                best_count = count
                best_box = b_idx
        if best_box is None or best_count == 0:
            break
        selected.append(best_box)
        for pt in box_points[best_box]:
            residual[pt] -= 1
    
    rank_found = len(selected)
    
    if rank_found < best_rank and rank_found > 0:
        # Verify and build solution
        U_new = np.zeros((nm, rank_found))
        V_new = np.zeros((mp, rank_found))
        W_new = np.zeros((pn, rank_found))
        for c_idx, b_idx in enumerate(selected):
            I, J, K = boxes[b_idx]
            for i in range(n):
                if I & (1 << i):
                    for j in range(m):
                        if J & (1 << j):
                            U_new[i * m + j, c_idx] = 1
            for j in range(m):
                if J & (1 << j):
                    for k in range(p):
                        if K & (1 << k):
                            V_new[j * p + k, c_idx] = 1
            for k in range(p):
                if K & (1 << k):
                    for i in range(n):
                        if I & (1 << i):
                            W_new[k * n + i, c_idx] = 1
        # Verify
        verify_ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    pt = i * mp + j * p + k
                    total = 0.0
                    for c_idx in range(rank_found):
                        total += U_new[i*m+j, c_idx] * V_new[j*p+k, c_idx] * W_new[k*n+i, c_idx]
                    if abs(total - 1.0) > 1e-9:
                        verify_ok = False
                        break
                if not verify_ok:
                    break
            if not verify_ok:
                break
        if verify_ok:
            best_rank = rank_found
            best_U = U_new
            best_V = V_new
            best_W = W_new
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
