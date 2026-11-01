import time

DIMENSIONS = [2, 4, 5]
START_TIME = time.time()
DEADLINE = 110.0

def check_deadline():
    return time.time() - START_TIME > DEADLINE

# EVOLVE-BLOCK-START
def construct():
    if check_deadline():
        # Return a trivial valid decomposition (rank 40? or just fail fast?)
        # If we fail fast, score is 0.
        # Better to return a known valid decomposition if we have time.
        # But if we are past deadline, we can't compute.
        # The directive says "return the best result found".
        # If we haven't computed, we can return a pre-computed one?
        # I will just let it run, it's fast.
        pass

    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def generate_strassen_222(u_base, v_base, w_base):
        terms = []
        # P1
        u1 = [(u_base[0], 1), (u_base[3], 1)]
        v1 = [(v_base[0], 1), (v_base[3], 1)]
        w1 = [(w_base[0], 1), (w_base[3], 1)]
        terms.append((u1, v1, w1))
        
        # P2
        u2 = [(u_base[2], 1), (u_base[3], 1)]
        v2 = [(v_base[0], 1)]
        w2 = [(w_base[2], 1)]
        terms.append((u2, v2, w2))
        
        # P3
        u3 = [(u_base[0], 1)]
        v3 = [(v_base[1], 1), (v_base[3], -1)]
        # FIX: Add w_base[3]
        w3 = [(w_base[1], 1), (w_base[3], 1)]
        terms.append((u3, v3, w3))
        
        # P4
        u4 = [(u_base[3], 1)]
        v4 = [(v_base[2], 1), (v_base[0], -1)]
        w4 = [(w_base[0], 1), (w_base[2], 1)]
        terms.append((u4, v4, w4))
        
        # P5
        u5 = [(u_base[0], 1), (u_base[1], 1)]
        v5 = [(v_base[3], 1)]
        w5 = [(w_base[1], 1), (w_base[0], -1)]
        terms.append((u5, v5, w5))
        
        # P6
        u6 = [(u_base[2], 1), (u_base[0], -1)]
        v6 = [(v_base[0], 1), (v_base[1], 1)]
        w6 = [(w_base[3], 1)]
        terms.append((u6, v6, w6))
        
        # P7
        u7 = [(u_base[1], 1), (u_base[3], -1)]
        v7 = [(v_base[2], 1), (v_base[3], 1)]
        w7 = [(w_base[0], 1), (w_base[3], 1)]
        terms.append((u7, v7, w7))
        
        return terms

    def add_2x2x1(u_base, v_base, w_base):
        terms = []
        terms.append([[(u_base[0], 1)], [(v_base[0], 1)], [(w_base[0], 1)]])
        terms.append([[(u_base[1], 1)], [(v_base[1], 1)], [(w_base[0], 1)]])
        terms.append([[(u_base[2], 1)], [(v_base[0], 1)], [(w_base[1], 1)]])
        terms.append([[(u_base[3], 1)], [(v_base[1], 1)], [(w_base[1], 1)]])
        return terms

    terms_L = []
    terms_L += generate_strassen_222([0,1,2,3], [0,1,5,6], [0,1,2,3])
    terms_L += generate_strassen_222([0,1,2,3], [2,3,7,8], [4,5,6,7])
    terms_L += add_2x2x1([0,1,2,3], [4,9], [8,9])
    
    terms_R = []
    terms_R += generate_strassen_222([4,5,6,7], [10,11,15,16], [0,1,2,3])
    terms_R += generate_strassen_222([4,5,6,7], [12,13,17,18], [4,5,6,7])
    terms_R += add_2x2x1([4,5,6,7], [14,19], [8,9])
    
    all_terms = terms_L + terms_R
    
    for idx, (u_part, v_part, w_part) in enumerate(all_terms):
        for u_idx, coeff in u_part:
            U[u_idx][idx] += coeff
        for v_idx, coeff in v_part:
            V[v_idx][idx] += coeff
        for w_idx, coeff in w_part:
            W[w_idx][idx] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
