"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10 second safety margin before 60s limit
    
    n = 30
    
    # 1. Calculate LCM of 1..30
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, n + 1))
    
    # 2. Calculate weights W_i = L // i
    weights = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    # 3. LLL Lattice Reduction
    # Lattice in Z^{n+1}: basis b_i = (e_i, w_i) for i=0..n-1
    # Vector: (x_0, ..., x_{n-1}, sum x_i * w_i)
    # We want x_i in {-1,0,1}, sum != 0, minimize |sum|
    
    # Build basis matrix (n x (n+1))
    basis = np.zeros((n, n + 1), dtype=np.int64)
    for i in range(n):
        basis[i, i] = 1
        basis[i, n] = weights[i]
    
    # Apply LLL reduction
    # LLL parameters: delta = 0.75, eta = 0.5
    def lll_reduce(B, delta=0.75, eta=0.5):
        """
        LLL basis reduction for integer matrix B (m x n).
        Returns reduced basis.
        """
        m, n = B.shape
        B = B.astype(np.float64).copy()
        
        # Gram-Schmidt
        def gram_schmidt(B):
            m, n = B.shape
            mu = np.zeros((m, m), dtype=np.float64)
            b_prime = np.zeros_like(B)
            for i in range(m):
                b_prime[i] = B[i].copy()
                for j in range(i):
                    if np.dot(b_prime[j], b_prime[j]) > 0:
                        mu[i, j] = np.dot(B[i], b_prime[j]) / np.dot(b_prime[j], b_prime[j])
                        b_prime[i] = b_prime[i] - mu[i, j] * b_prime[j]
            return mu, b_prime
        
        # Size-reduce
        def size_reduce(B, mu):
            m, n = B.shape
            for k in range(1, m):
                for j in range(k - 1, -1, -1):
                    q = round(mu[k, j])
                    B[k] = B[k] - q * B[j]
                    # Update mu
                    for i in range(k):
                        if i != j:
                            mu[k, i] = mu[k, i] - q * mu[j, i]
                    mu[k, j] = mu[k, j] - q
            return B, mu
        
        # LLL main loop
        mu, b_prime = gram_schmidt(B)
        k = 1
        while k < m:
            # Size-reduce
            B, mu = size_reduce(B, mu)
            
            # Check Lovasz condition
            if delta * np.dot(b_prime[k-1], b_prime[k-1]) <= np.dot(b_prime[k], b_prime[k]) - np.dot(b_prime[k-1], b_prime[k-1]) * mu[k, k-1] ** 2:
                k += 1
                if k < m:
                    mu, b_prime = gram_schmidt(B)
            else:
                # Swap
                B[k-1], B[k] = B[k].copy(), B[k-1].copy()
                # Update Gram-Schmidt
                mu, b_prime = gram_schmidt(B)
                k = max(0, k - 1)
        
        return B.astype(np.int64)
    
    try:
        reduced_basis = lll_reduce(basis)
    except:
        reduced_basis = basis
    
    # 4. Extract candidate solutions from reduced basis
    # Each row of reduced_basis is (x_0, ..., x_{n-1}, s) where s = sum x_i * w_i
    # We want x_i in {-1,0,1} and s != 0, minimize |s|
    
    best_coeffs = None
    best_abs_sum = np.iinfo(np.int64).max
    
    for row in reduced_basis:
        x = row[:n]
        s = row[n]
        
        # Check if x_i are in {-1, 0, 1}
        if np.all((x >= -1) & (x <= 1)):
            if s != 0:
                abs_s = abs(s)
                if abs_s < best_abs_sum:
                    best_abs_sum = abs_s
                    best_coeffs = x.tolist()
    
    # 5. If LLL didn't find a good solution, use meet-in-the-middle
    if best_coeffs is None or time.time() > deadline - 20:
        mid = n // 2
        w1 = weights[:mid]
        w2 = weights[mid:]
        
        def generate_sums_with_indices(ws):
            sums = np.array([0], dtype=np.int64)
            indices = np.array([0], dtype=np.int32)
            for w in ws:
                n_curr = len(sums)
                new_sums = np.empty(3 * n_curr, dtype=np.int64)
                new_indices = np.empty(3 * n_curr, dtype=np.int32)
                
                new_sums[0:n_curr] = sums - w
                new_indices[0:n_curr] = indices * 3 + 0
                
                new_sums[n_curr:2*n_curr] = sums
                new_indices[n_curr:2*n_curr] = indices * 3 + 1
                
                new_sums[2*n_curr:3*n_curr] = sums + w
                new_indices[2*n_curr:3*n_curr] = indices * 3 + 2
                
                sums = new_sums
                indices = new_indices
            return sums, indices
        
        sums1, idx1 = generate_sums_with_indices(w1)
        sums2, idx2 = generate_sums_with_indices(w2)
        
        if time.time() > deadline - 10:
            return [0] * 28 + [1, -1]
        
        sort_order = np.argsort(sums2)
        sorted_sums2 = sums2[sort_order]
        sorted_idx2 = idx2[sort_order]
        
        neg_sums1 = -sums1
        insert_pos = np.searchsorted(sorted_sums2, neg_sums1, side='left')
        
        n_sums2 = len(sorted_sums2)
        
        pos1 = np.clip(insert_pos, 0, n_sums2 - 1)
        pos2 = np.clip(insert_pos - 1, 0, n_sums2 - 1)
        
        cand1 = sorted_sums2[pos1]
        cand2 = sorted_sums2[pos2]
        
        total1 = sums1 + cand1
        total2 = sums1 + cand2
        
        abs_total1 = np.where(total1 != 0, np.abs(total1), np.iinfo(np.int64).max)
        abs_total2 = np.where(total2 != 0, np.abs(total2), np.iinfo(np.int64).max)
        
        best_abs = np.minimum(abs_total1, abs_total2)
        best_pos = np.where(abs_total1 <= abs_total2, pos1, pos2)
        
        min_abs = np.min(best_abs)
        
        if min_abs == np.iinfo(np.int64).max:
            return [0] * 28 + [1, -1]
        
        k1_idx = np.argmin(best_abs)
        pos_in_sorted = best_pos[k1_idx]
        orig_pos = sort_order[pos_in_sorted]
        idx_in_sums2 = orig_pos
        idx_in_sums1 = k1_idx
        
        def decode_index(k, n_weights):
            coeffs = []
            for _ in range(n_weights):
                digit = k % 3
                if digit == 0:
                    coeffs.append(-1)
                elif digit == 1:
                    coeffs.append(0)
                else:
                    coeffs.append(1)
                k //= 3
            coeffs.reverse()
            return coeffs
        
        coeffs1 = decode_index(idx1[idx_in_sums1], len(w1))
        coeffs2 = decode_index(idx2[idx_in_sums2], len(w2))
        
        result = coeffs1 + coeffs2
    else:
        result = best_coeffs
    
    # Verify
    actual_sum = 0
    for i, c in enumerate(result):
        if c != 0:
            actual_sum += c * (L // (i + 1))
    
    if actual_sum == 0:
        return [0] * 28 + [1, -1]
    
    return result

# EVOLVE-BLOCK-END
