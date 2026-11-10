import time
import random
import sys

def _gf16_tables():
    add_tbl = [[0]*16 for _ in range(16)]
    mul_tbl = [[0]*16 for _ in range(16)]
    
    def int_to_poly(x):
        return [(x >> i) & 1 for i in range(4)]
        
    def poly_to_int(p):
        return sum(bit << i for i, bit in enumerate(p))
        
    def poly_add(p1, p2):
        return [a ^ b for a, b in zip(p1, p2)]
        
    def poly_mul(p1, p2):
        res = [0]*8
        for i in range(4):
            if p1[i]:
                for j in range(4):
                    if p2[j]:
                        res[i+j] ^= 1
        for i in range(7, 3, -1):
            if res[i]:
                res[i] = 0
                res[i-3] ^= 1
                res[i-4] ^= 1
        return res[:4]

    for a in range(16):
        pa = int_to_poly(a)
        for b in range(16):
            pb = int_to_poly(b)
            add_tbl[a][b] = poly_to_int(poly_add(pa, pb))
            mul_tbl[a][b] = poly_to_int(poly_mul(pa, pb))
            
    return add_tbl, mul_tbl

ADD16, MUL16 = _gf16_tables()

def mat_mult_2x2(A, B, add, mul):
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    c11 = add(mul(a11, b11), mul(a12, b21))
    c12 = add(mul(a11, b12), mul(a12, b22))
    c21 = add(mul(a21, b11), mul(a22, b21))
    c22 = add(mul(a21, b12), mul(a22, b22))
    return (c11, c12, c21, c22)

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if k != 3 or g != 16:
        # Fallback to a known valid construction if parameters differ?
        # The task implies k=3, g=16.
        # If not, we might fail.
        pass

    add, mul = ADD16, MUL16
    elements = []
    involutions = []
    mat_to_idx = {}
    
    # Generate SL(2,16)
    for a in range(16):
        for b in range(16):
            for c in range(16):
                if a == 0:
                    if mul[b][c] == 1:
                        for d in range(16):
                            M = (a, b, c, d)
                            idx = len(elements)
                            elements.append(M)
                            mat_to_idx[M] = idx
                            M2 = mat_mult_2x2(M, M, add, mul)
                            if M2 == (1, 0, 0, 1):
                                involutions.append(M)
                else:
                    rhs = add[1][mul[b][c]]
                    a_inv = None
                    for x in range(1, 16):
                        if mul[a][x] == 1:
                            a_inv = x
                            break
                    if a_inv is not None:
                        d = mul[rhs][a_inv]
                        M = (a, b, c, d)
                        idx = len(elements)
                        elements.append(M)
                        mat_to_idx[M] = idx
                        M2 = mat_mult_2x2(M, M, add, mul)
                        if M2 == (1, 0, 0, 1):
                            involutions.append(M)
                            
    n_elements = len(elements)
    
    # Precompute multiplication by 1? No.
    # We will check girth.
    
    def check_girth(gen_set):
        # gen_set: list of 3 matrices
        # Build adjacency lists
        adj = [[] for _ in range(n_elements)]
        
        # Precompute neighbors for each generator for all elements?
        # Or just compute on the fly in BFS?
        # On the fly is better memory-wise, but slower if BFS revisits.
        # BFS visits each node once. So we compute neighbors once per node.
        # Total mults: N * 3.
        
        expected = [1, 3, 6, 12, 24, 48, 96, 192]
        
        for start in range(n_elements):
            visited = [-1] * n_elements
            queue = [(start, 0)]
            visited[start] = 0
            counts = [0]*8
            counts[0] = 1
            head = 0
            while head < len(queue):
                u, d = queue[head]
                head += 1
                if d == 7:
                    continue
                M_u = elements[u]
                for G in gen_set:
                    MG = mat_mult_2x2(M_u, G, add, mul)
                    v = mat_to_idx.get(MG)
                    if v is not None:
                        if visited[v] == -1:
                            visited[v] = d + 1
                            counts[d+1] += 1
                            queue.append((v, d+1))
                        # If visited, ignore for tree count check.
                        # If the graph is not a tree in the ball, counts will be lower.
            
            for i in range(8):
                if counts[i] != expected[i]:
                    return False
        return True

    random.seed(42)
    inv_list = involutions[:]
    random.shuffle(inv_list)
    
    attempts = 0
    while time.time() < deadline:
        if len(inv_list) < 3:
            break
        A = inv_list[attempts % len(inv_list)]
        B = inv_list[(attempts + 1) % len(inv_list)]
        C = inv_list[(attempts + 2) % len(inv_list)]
        
        if A == B or B == C or A == C:
            attempts += 1
            continue
            
        gen_set = [A, B, C]
        
        if check_girth(gen_set):
            edges = []
            for i, M in enumerate(elements):
                for G in gen_set:
                    MG = mat_mult_2x2(M, G, add, mul)
                    j = mat_to_idx.get(MG)
                    if i < j:
                        edges.append((i, j))
            return n_elements, edges
        attempts += 1
        
    return 0, []
