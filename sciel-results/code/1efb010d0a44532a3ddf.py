import time
import random

def construct():
    deadline = time.time() + 110  # Safety margin under 120s kill
    
    # Construct the Schläfli graph: SRG(35, 16, 7, 8)
    # Vertices: (a, b) where a in Z_5, b in Z_7
    # Two vertices (a1,b1) and (a2,b2) are adjacent iff:
    #   (a2-a1)^2 * (b2-b1)^2 mod 5 is NOT a quadratic residue (i.e., = 4 mod 5)
    #   AND the difference (b2-b1) mod 7 is nonzero
    # Actually, let me use the well-known construction:
    # The Schläfli graph can be constructed as follows:
    # Label vertices by (i, j) with i in {0,...,4}, j in {0,...,6}
    # (i,j) ~ (k,l) iff (k-i) mod 5 != 0 and ((l-j) mod 7)^2 * ((k-i) mod 5)^2 mod 7 is a quadratic residue mod 7
    # 
    # Let me use a cleaner known construction. The Schläfli graph is the complement of the 
    # point graph of the generalized quadrangle GQ(2,4)? No.
    #
    # Actually, the Schläfli graph is a well-known SRG(35,16,7,8). Let me use the 
    # standard construction via the Paley-like approach or direct edge list.
    #
    # A reliable construction: The Schläfli graph has vertices (x,y) in Z_5 x Z_7.
    # (x1,y1) adjacent to (x2,y2) iff (x2-x1) mod 5 != 0 and (y2-y1)^2 * (x2-x1)^(-2) mod 7 is in {1, 2, 4} (quadratic residues mod 7).
    #
    # Wait, let me verify: quadratic residues mod 7 are {1, 2, 4}.
    # For each pair (x1,y1),(x2,y2) with x1 != x2:
    #   d = (x2-x1) mod 5, we need d != 0
    #   We compute (y2-y1)^2 / d^2 mod 7. Since d is in {1,2,3,4}, d^2 mod 7 is in {1,4,2,2}
    #   Actually d mod 5, but we're working mod 7 for the ratio.
    #   Let me just compute: for d in 1..4, d^2 mod 7 = 1, 4, 2, 2
    #   So (y2-y1)^2 * (d^2)^(-1) mod 7 should be a QR mod 7.
    #   Inverse of 1 is 1, inverse of 4 is 2 (4*2=8=1), inverse of 2 is 4 (2*4=8=1)
    #   So: if d^2 mod 7 == 1: check (y2-y1)^2 mod 7 in {1,2,4}
    #        if d^2 mod 7 == 4: check (y2-y1)^2 * 2 mod 7 in {1,2,4}
    #        if d^2 mod 7 == 2: check (y2-y1)^2 * 4 mod 7 in {1,2,4}
    #
    # Let me just code it directly.
    
    n = 35
    # Map (i,j) -> vertex index
    def idx(i, j):
        return i * 7 + j
    
    # Build adjacency
    A = [0] * n
    
    for i1 in range(5):
        for j1 in range(7):
            for i2 in range(5):
                for j2 in range(7):
                    if i1 == i2:
                        continue
                    d = (i2 - i1) % 5
                    if d == 0:
                        continue
                    dy = (j2 - j1) % 7
                    if dy == 0:
                        continue
                    # Compute (dy^2) * (d^2)^(-1) mod 7
                    dy2 = (dy * dy) % 7
                    d2 = (d * d) % 7
                    # Inverse of d2 mod 7
                    inv_d2 = pow(d2, 6, 7)  # Fermat's little theorem: d2^6 = 1 mod 7
                    ratio = (dy2 * inv_d2) % 7
                    if ratio in (1, 2, 4):
                        v1 = idx(i1, j1)
                        v2 = idx(i2, j2)
                        A[v1] |= (1 << v2)
                        A[v2] |= (1 << v1)
    
    # Verify it's a valid graph: no K4, no independent set of size 6
    def count_k4(n, A):
        count = 0
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
                    c_bit = mask_c & -c_bit
                    c = c_bit.bit_length() - 1
                    mask_c ^= c_bit
                    nc = A[c] & A[a] & A[b] & ~((1 << (c + 1)) - 1)
                    count += bin(nc).count('1')
        return count
    
    def max_independent_set_size(n, A):
        comp = [0] * n
        all_mask = (1 << n) - 1
        for i in range(n):
            comp[i] = all_mask & ~(A[i] | (1 << i))
        
        best = [0]
        
        def bk(R, P, X):
            if not P:
                if len(R) > best[0]:
                    best[0] = len(R)
                return
            if len(R) + bin(P).count('1') <= best[0]:
                return
            
            if X:
                u_bit = X & -X
                u = u_bit.bit_length() - 1
                candidates = P & ~comp[u]
            else:
                max_deg = -1
                u = -1
                p_temp = P
                while p_temp:
                    v_bit = p_temp & -p_temp
                    v = v_bit.bit_length() - 1
                    deg = bin(comp[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                    p_temp ^= v_bit
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
        return best[0]
    
    # Check K4-free
    k4_count = count_k4(n, A)
    if k4_count > 0:
        # Fallback: use Paley 17 (shouldn't happen with correct Schläfli construction)
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
    
    # Check independence number
    alpha = max_independent_set_size(n, A)
    if alpha >= 6:
        # Fallback: use Paley 17
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        A = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % 17 in squares:
                    A[i] |= (1 << j)
                    A[j] |= (1 << i)
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (A[j] >> i) & 1:
                edges.append([i, j])
    
    return {'n': n, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
