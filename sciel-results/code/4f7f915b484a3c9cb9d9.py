import time
import numpy as np

DEADLINE = 120.0
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START)

def _build_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # P1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (A21+A22)A11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # P3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # P4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # P5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[3,4]=1
    # P6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _embed(t_U, t_V, t_W, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
    """Embed a small tensor into a larger space."""
    rank = t_U.shape[1]
    U = np.zeros((n_tgt * m_tgt, rank))
    V = np.zeros((m_tgt * p_tgt, rank))
    W = np.zeros((p_tgt * n_tgt, rank))
    for i in range(n_src):
        for j in range(m_src):
            U[n_map[i]*m_tgt + m_map[j], :] = t_U[i*m_src+j, :]
    for j in range(m_src):
        for k in range(p_src):
            V[m_map[j]*p_tgt + p_map[k], :] = t_V[j*p_src+k, :]
    for k in range(p_src):
        for i in range(n_src):
            W[p_map[k]*n_tgt + n_map[i], :] = t_W[k*n_src+i, :]
    return U, V, W

def _tensor_sum(U_list, V_list, W_list):
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _build_2x2x3():
    """
    Build 2x2x3 decomposition.
    Rank = Rank(2x2x2) + Rank(2x2x1) = 7 + 2 = 9?
    Schoolbook 2x2x1 has rank 2.
    Total 9.
    """
    S_U, S_V, S_W = _build_strassen_222()
    sch_U, sch_V, sch_W = _build_schoolbook(2, 2, 1)
    
    # Embed Strassen into p=[0,1]
    u_s, v_s, w_s = _embed(S_U, S_V, S_W, 2,2,2, 2,2,3, [0,1],[0,1],[0,1])
    # Embed Schoolbook into p=[2]
    u_sc, v_sc, w_sc = _embed(sch_U, sch_V, sch_W, 2,2,1, 2,2,3, [0,1],[0,1],[2])
    
    U = np.hstack([u_s, u_sc])
    V = np.hstack([v_s, v_sc])
    W = np.hstack([w_s, w_sc])
    return U, V, W

def _build_2x2x4():
    """
    Build 2x2x4 decomposition.
    Rank = 2 * Rank(2x2x2) = 14.
    """
    S_U, S_V, S_W = _build_strassen_222()
    
    u1, v1, w1 = _embed(S_U, S_V, S_W, 2,2,2, 2,2,4, [0,1],[0,1],[0,1])
    u2, v2, w2 = _embed(S_U, S_V, S_W, 2,2,2, 2,2,4, [0,1],[0,1],[2,3])
    
    U = np.hstack([u1, u2])
    V = np.hstack([v1, v2])
    W = np.hstack([w1, w2])
    return U, V, W

def _build_2x4x7():
    """
    Build 2x4x7 decomposition using blocks.
    Split m=4 into 2,2.
    Split p=7 into 3,4.
    
    Block 1: 2x2x3 (m=0,1; p=0,1,2) -> Rank 9
    Block 2: 2x2x4 (m=0,1; p=3,4,5,6) -> Rank 14
    Block 3: 2x2x3 (m=2,3; p=0,1,2) -> Rank 9
    Block 4: 2x2x4 (m=2,3; p=3,4,5,6) -> Rank 14
    
    Total Rank = 9 + 14 + 9 + 14 = 46.
    """
    U_parts, V_parts, W_parts = [], [], []
    
    # Block 1: m=[0,1], p=[0,1,2]
    U1, V1, W1 = _build_2x2x3()
    u1, v1, w1 = _embed(U1, V1, W1, 2,2,3, 2,4,7, [0,1], [0,1], [0,1,2])
    U_parts.append(u1); V_parts.append(v1); W_parts.append(w1)
    
    # Block 2: m=[0,1], p=[3,4,5,6]
    U2, V2, W2 = _build_2x2x4()
    u2, v2, w2 = _embed(U2, V2, W2, 2,2,4, 2,4,7, [0,1], [0,1], [3,4,5,6])
    U_parts.append(u2); V_parts.append(v2); W_parts.append(w2)
    
    # Block 3: m=[2,3], p=[0,1,2]
    U3, V3, W3 = _build_2x2x3()
    u3, v3, w3 = _embed(U3, V3, W3, 2,2,3, 2,4,7, [0,1], [2,3], [0,1,2])
    U_parts.append(u3); V_parts.append(v3); W_parts.append(w3)
    
    # Block 4: m=[2,3], p=[3,4,5,6]
    U4, V4, W4 = _build_2x2x4()
    u4, v4, w4 = _embed(U4, V4, W4, 2,2,4, 2,4,7, [0,1], [2,3], [3,4,5,6])
    U_parts.append(u4); V_parts.append(v4); W_parts.append(w4)
    
    return _tensor_sum(U_parts, V_parts, W_parts)

def construct():
    t0 = time.time()
    
    # Use the block decomposition (Rank 46)
    U, V, W = _build_2x4x7()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
