import time
import random
import sys

def construct():
    deadline = time.time() + 110
    rng = random.Random(42)
    
    # Helper: Exact K4 count
    def count_k4(n, A):
        count = 0
        for a in range(n):
            na = A[a]
            # Iterate over neighbors b of a with b > a
            mask_b = na & ~((1 << (a + 1)) - 1)
            while mask_b:
                b_bit = mask_b & -mask_b
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                
                # Common neighbors of a and b, with c > b
                nab = A[b] & A[a] & ~((1 << (b + 1)) - 1)
                mask_c = nab
                while mask_c:
                    c_bit = mask_c & -mask_c
                    c = c_bit.bit_length() - 1
                    mask_c ^= c_bit
                    
                    # Common neighbors of a, b, c, with d > c
                    nc = A[c] & A[a] & A[b] & ~((1 << (c + 1)) - 1)
                    count += bin(nc).count('1')
        return count

    # Helper: Find a specific K4
    def find_k4(n, A):
        for a in range(n):
            na = A[a] & ~((1 << (a + 1)) - 1)
            mask_b = na
            while mask_b:
                b_bit = mask_b & -mask_b
                b = b_bit.bit_length() - 1
                mask_b ^= b_bit
                nab = A[b] & A[a] & ~((1 << (b + 1)) - 1)
                mask_c = nab
                while mask_c:
                    c_bit = mask_c & -mask_c
                    c = c_bit.bit_length() - 1
                    mask_c ^= c_bit
                    nc = A[c] & A[a] & A[b] & ~((1 << (c + 1)) - 1)
                    if nc:
                        d_bit = nc & -nc
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    # Helper: Check for Independent Set of size 6
    def has_ind6(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [False]
        
        def bk(R, P, X):
            if found[0]:
                return
            if len(R) == 6:
                found[0] = True
                return
            if not P:
                return
            if len(R) + bin(P).count('1') < 6:
                return
            
            # Pivot
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
                
            c = candidates
            while c:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    # Helper: Find an independent set of size 6
    def find_ind6(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        found = [None]
        
        def bk(R, P, X):
            if found[0] is not None:
                return
            if len(R) == 6:
                found[0] = tuple(R)
                return
            if not P:
                return
            if len(R) + bin(P).count('1') < 6:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                candidates = P
                
            c = candidates
            while c and found[0] is None:
                low = c & -c
                v = low.bit_length() - 1
                bk(R + [v], P & comp[v], X & comp[v])
                P = P & ~low
                X = X | low
                c = c & ~low
        
        bk([], all_mask, 0)
        return found[0]

    # Main Logic
    best_n = 0
    best_edges = []
    
    # Try to construct n=35
    for target_n in [35, 34, 33, 32]:
        if time.time() > deadline - 20:
            break
            
        n = target_n
        # Initialize with a random graph of density ~0.4
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.4:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
        
        # Local Repair
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Check K4
            k4 = find_k4(n, A)
            if k4:
                a, b, c, d = k4
                edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                u, v = rng.choice(edges)
                A[u] &= ~(1 << v)
                A[v] &= ~(1 << u)
                continue
            
            # Check Ind6
            ind6 = find_ind6(n, A)
            if ind6:
                verts = list(ind6)
                u, v = rng.sample(verts, 2)
                A[u] |= (1 << v)
                A[v] |= (1 << u)
                continue
            
            # Valid
            best_n = n
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (A[j] >> i) & 1:
                        edges.append([i, j])
            best_edges = edges
            break
        
        if best_n == target_n:
            break

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}")
