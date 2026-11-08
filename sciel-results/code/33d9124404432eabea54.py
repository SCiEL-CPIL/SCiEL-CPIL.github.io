"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Outer Strassen combinations
    # X_alpha = sum c * A_block
    # Y_alpha = sum c * B_block
    outer_X = [
        {11: 1, 22: 1},
        {21: 1, 22: 1},
        {11: 1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 11: -1},
        {12: 1, 22: -1}
    ]
    outer_Y = [
        {11: 1, 22: 1},
        {11: 1},
        {12: 1, 22: -1},
        {21: 1, 11: -1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 22: 1}
    ]
    
    # Inner Strassen combinations
    inner_X = [
        {11: 1, 22: 1},
        {21: 1, 22: 1},
        {11: 1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 11: -1},
        {12: 1, 22: -1}
    ]
    inner_Y = [
        {11: 1, 22: 1},
        {11: 1},
        {12: 1, 22: -1},
        {21: 1, 11: -1},
        {22: 1},
        {11: 1, 12: 1},
        {21: 1, 22: 1}
    ]
    
    # Coefficients for C blocks from P_alpha
    C_coeff = [
        {11: 1, 12: 0, 21: 0, 22: 1}, # P_1
        {11: 0, 12: 0, 21: 1, 22: -1}, # P_2
        {11: 0, 12: 1, 21: 0, 22: 1}, # P_3
        {11: 1, 12: 0, 21: 1, 22: 0}, # P_4
        {11: -1, 12: 1, 21: 0, 22: 0}, # P_5
        {11: 0, 12: 0, 21: 0, 22: 1}, # P_6
        {11: 1, 12: 0, 21: 0, 22: 0}  # P_7
    ]
    
    # Inner C coefficients
    inner_C_coeff = [
        {11: 1, 12: 0, 21: 0, 22: 1}, # m_1
        {11: 0, 12: 0, 21: 1, 22: -1}, # m_2
        {11: 0, 12: 1, 21: 0, 22: 1}, # m_3
        {11: 1, 12: 0, 21: 1, 22: 0}, # m_4
        {11: -1, 12: 1, 21: 0, 22: 0}, # m_5
        {11: 0, 12: 0, 21: 0, 22: 1}, # m_6
        {11: 1, 12: 0, 21: 0, 22: 0}  # m_7
    ]
    
    for alpha in range(7):
        for beta in range(7):
            r = alpha * 7 + beta
            
            U_vec = [0.0] * 16
            V_vec = [0.0] * 16
            W_vec = [0.0] * 16
            
            # Compute U_vec
            for e_idx, e_coeff in inner_X[beta].items():
                if e_idx == 11: r_e, c_e = 0, 0
                elif e_idx == 12: r_e, c_e = 0, 1
                elif e_idx == 21: r_e, c_e = 1, 0
                else: r_e, c_e = 1, 1
                
                for b_idx, b_coeff in outer_X[alpha].items():
                    if b_idx == 11: br, bc = 0, 0
                    elif b_idx == 12: br, bc = 0, 1
                    elif b_idx == 21: br, bc = 1, 0
                    else: br, bc = 1, 1
                    
                    i = 2 * br + r_e
                    j = 2 * bc + c_e
                    U_vec[i * 4 + j] += e_coeff * b_coeff
            
            # Compute V_vec
            for e_idx, e_coeff in inner_Y[beta].items():
                if e_idx == 11: r_e, c_e = 0, 0
                elif e_idx == 12: r_e, c_e = 0, 1
                elif e_idx == 21: r_e, c_e = 1, 0
                else: r_e, c_e = 1, 1
                
                for b_idx, b_coeff in outer_Y[alpha].items():
                    if b_idx == 11: br, bc = 0, 0
                    elif b_idx == 12: br, bc = 0, 1
                    elif b_idx == 21: br, bc = 1, 0
                    else: br, bc = 1, 1
                    
                    j = 2 * br + r_e
                    k = 2 * bc + c_e
                    V_vec[j * 4 + k] += e_coeff * b_coeff
            
            # Compute W_vec
            for c_idx, c_coeff in C_coeff[alpha].items():
                if c_idx == 11: cr, cc = 0, 0
                elif c_idx == 12: cr, cc = 0, 1
                elif c_idx == 21: cr, cc = 1, 0
                else: cr, cc = 1, 1
                
                for p_idx, p_coeff in inner_C_coeff[beta].items():
                    if p_idx == 11: pr, pc = 0, 0
                    elif p_idx == 12: pr, pc = 0, 1
                    elif p_idx == 21: pr, pc = 1, 0
                    else: pr, pc = 1, 1
                    
                    i = 2 * cr + pr
                    k = 2 * cc + pc
                    W_vec[k * 4 + i] += c_coeff * p_coeff
            
            for idx in range(16):
                U[idx, r] = U_vec[idx]
                V[idx, r] = V_vec[idx]
                W[idx, r] = W_vec[idx]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
