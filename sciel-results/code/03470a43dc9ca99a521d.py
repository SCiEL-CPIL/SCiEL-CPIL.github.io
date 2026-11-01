import time
import sys

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # Rank 53: 7 blocks of 2x2 (Strassen) + 1 block of 2x1 (Schoolbook)
    # 7 * 7 + 4 = 53
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
        
    # Strassen block for 2x2 by 2x2
    # Indices:
    # U: i*m+j (0..3)
    # V: j*p+k (0..3 for local 2x2)
    # W: k*n+i (0..3 for local 2x2)
    
    def strassen_block(base_u, base_v, base_w):
        # base_u, base_v, base_w are offsets for the vectors
        # We add 7 terms
        # Term 1
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[3]=1
        add_term(u, v, w)
        
        # Term 2
        u = [0]*4; u[2]=1; u[3]=1
        v = [0]*4; v[0]=1
        w = [0]*4; w[1]=1
        add_term(u, v, w)
        
        # Term 3
        u = [0]*4; u[0]=1
        v = [0]*4; v[1]=1; v[3]=-1
        w = [0]*4; w[2]=1
        add_term(u, v, w)
        
        # Term 4
        u = [0]*4; u[3]=1
        v = [0]*4; v[2]=1; v[0]=-1
        w = [0]*4; w[1]=1
        add_term(u, v, w)
        
        # Term 5
        u = [0]*4; u[0]=1; u[1]=1
        v = [0]*4; v[3]=1
        w = [0]*4; w[3]=1
        add_term(u, v, w)
        
        # Term 6
        u = [0]*4; u[2]=1; u[0]=-1
        v = [0]*4; v[0]=1; v[1]=1
        w = [0]*4; w[0]=1; w[3]=1
        add_term(u, v, w)
        
        # Term 7
        u = [0]*4; u[1]=1; u[3]=-1
        v = [0]*4; v[2]=1; v[3]=1
        w = [0]*4; w[0]=-1; w[2]=1
        add_term(u, v, w)

    def schoolbook_block(base_u, base_v, base_w):
        # 2x2 by 2x1
        # 4 terms
        # c_0 = a_0 b_0 + a_1 b_1
        # c_1 = a_2 b_0 + a_3 b_1
        # U: 0..3, V: 0..1, W: 0..1
        
        # T1: a0 b0 -> c0
        u = [0]*4; u[0]=1
        v = [0]*2; v[0]=1
        w = [0]*2; w[0]=1
        add_term(u, v, w)
        
        # T2: a1 b1 -> c0
        u = [0]*4; u[1]=1
        v = [0]*2; v[1]=1
        w = [0]*2; w[0]=1
        add_term(u, v, w)
        
        # T3: a2 b0 -> c1
        u = [0]*4; u[2]=1
        v = [0]*2; v[0]=1
        w = [0]*2; w[1]=1
        add_term(u, v, w)
        
        # T4: a3 b1 -> c1
        u = [0]*4; u[3]=1
        v = [0]*2; v[1]=1
        w = [0]*2; w[1]=1
        add_term(u, v, w)

    # We need to map local indices to global indices
    # Global U size: 4
    # Global V size: 2*15 = 30
    # Global W size: 15*2 = 30
    
    # Clear U, V, W
    U = []
    V = []
    W = []
    
    # 7 Strassen blocks (cols 0-1, 2-3, ..., 12-13)
    for b in range(7):
        # Local V indices: 0,1,2,3 correspond to global j*15+k
        # j=0: k=2b, 2b+1 -> 0, 1
        # j=1: k=2b, 2b+1 -> 15, 16
        # So local v[0] -> global 2b
        # local v[1] -> global 2b+1
        # local v[2] -> global 15+2b
        # local v[3] -> global 15+2b+1
        
        # Local W indices: 0,1,2,3 correspond to global k*n+i
        # k=0: i=0,1 -> 0, 1
        # k=1: i=0,1 -> 2, 3
        # But global W index is k*2+i.
        # For block b, k is 2b and 2b+1.
        # local w[0] (k=0, i=0) -> global (2b)*2 + 0 = 4b
        # local w[1] (k=0, i=1) -> global (2b)*2 + 1 = 4b+1
        # local w[2] (k=1, i=0) -> global (2b+1)*2 + 0 = 4b+2
        # local w[3] (k=1, i=1) -> global (2b+1)*2 + 1 = 4b+3
        
        def map_u(local_u):
            return [local_u[i] for i in range(4)]
            
        def map_v(local_v):
            gv = [0]*30
            gv[2*b] = local_v[0]
            gv[2*b+1] = local_v[1]
            gv[15+2*b] = local_v[2]
            gv[15+2*b+1] = local_v[3]
            return gv
            
        def map_w(local_w):
            gw = [0]*30
            gw[4*b] = local_w[0]
            gw[4*b+1] = local_w[1]
            gw[4*b+2] = local_w[2]
            gw[4*b+3] = local_w[3]
            return gw

        # Save original add_term
        orig_add = add_term
        def add_term_mapped(u, v, w):
            orig_add(map_u(u), map_v(v), map_w(w))
        add_term = add_term_mapped
        
        strassen_block(0,0,0)
        
        # Restore
        add_term = orig_add

    # 1 Schoolbook block (col 14)
    # Local V: 0,1 -> global 14, 29
    # Local W: 0,1 -> global 28, 29
    
    def map_u_sb(local_u):
        return [local_u[i] for i in range(4)]
        
    def map_v_sb(local_v):
        gv = [0]*30
        gv[14] = local_v[0]
        gv[29] = local_v[1]
        return gv
        
    def map_w_sb(local_w):
        gw = [0]*30
        gw[28] = local_w[0]
        gw[29] = local_w[1]
        return gw

    orig_add = add_term
    def add_term_sb(u, v, w):
        orig_add(map_u_sb(u), map_v_sb(v), map_w_sb(w))
    add_term = add_term_sb
    
    schoolbook_block(0,0,0)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'])}")
