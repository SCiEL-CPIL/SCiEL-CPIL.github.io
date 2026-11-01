"""Program for 2x2 x 2x14 matrix multiplication rank minimization."""
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition using Strassen's algorithm applied to 7 blocks of 2 columns.
    We group the 14 columns of B into 7 blocks of 2 columns each.
    For each block, we use Strassen's 7 products to compute the 2x2 result block.
    The 7 linear combinations of A are shared across all 7 blocks.
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For 2x2 x 2x14, we have 7 blocks of 2 columns.
    # For block s (s=0..6), B_block has columns 2s and 2s+1.
    # Let B_s = [[b_{0,2s}, b_{0,2s+1}], [b_{1,2s}, b_{1,2s+1}]]
    # The Strassen products for block s are:
    # m1_s = (a00 + a11)(b_{0,2s} + b_{1,2s+1})  -- wait, let me re-derive
    
    # Actually, let me be more careful.
    # A is 2x2: A = [[a00, a01], [a10, a11]]
    # B is 2x14, grouped into 7 blocks of 2x2.
    # Block s: B_s = [[b_{0,2s}, b_{0,2s+1}], [b_{1,2s}, b_{1,2s+1}]]
    # C_s = A * B_s = [[c_{0,2s}, c_{0,2s+1}], [c_{1,2s}, c_{1,2s+1}]]
    
    # Strassen for block s:
    # m1_s = (a00 + a11) * (b_{0,2s} + b_{1,2s+1})
    # m2_s = (a10 + a11) * b_{0,2s}
    # m3_s = a00 * (b_{0,2s+1} - b_{1,2s+1})
    # m4_s = a11 * (b_{1,2s} - b_{0,2s})
    # m5_s = (a00 + a01) * b_{1,2s+1}
    # m6_s = (a10 - a00) * (b_{0,2s} + b_{0,2s+1})
    # m7_s = (a01 - a11) * (b_{1,2s} + b_{1,2s+1})
    
    # c_{0,2s} = m1_s + m4_s - m5_s + m7_s
    # c_{0,2s+1} = m3_s + m5_s
    # c_{1,2s} = m2_s + m4_s
    # c_{1,2s+1} = m1_s - m2_s + m3_s + m6_s
    
    # Now, the U, V, W matrices:
    # U has shape (n*m, rank) = (4, 49)
    # V has shape (m*p, rank) = (28, 49)
    # W has shape (p*n, rank) = (28, 49)
    
    # The indices:
    # U: row = i*m + j = i*2 + j, where i in {0,1}, j in {0,1}
    # V: row = j*p + k = j*14 + k, where j in {0,1}, k in {0..13}
    # W: row = k*n + i = k*2 + i, where k in {0..13}, i in {0,1}
    
    # For each of the 49 products (7 blocks x 7 Strassen products), we have one column in U, V, W.
    
    rank = 49
    U = [[Fraction(0)] * rank for _ in range(4)]
    V = [[Fraction(0)] * rank for _ in range(28)]
    W = [[Fraction(0)] * rank for _ in range(28)]
    
    # A indices: a00=0, a01=1, a10=2, a11=3 (row = i*2+j)
    # a00 -> U[0], a01 -> U[1], a10 -> U[2], a11 -> U[3]
    
    # For block s, product t (t=0..6):
    # col = s*7 + t
    
    for s in range(7):
        k0 = 2 * s      # first column of block
        k1 = 2 * s + 1  # second column of block
        
        # t=0: m1_s = (a00 + a11) * (b_{0,k0} + b_{1,k1})
        col = s * 7 + 0
        U[0][col] = Fraction(1)  # a00
        U[3][col] = Fraction(1)  # a11
        V[0 * 14 + k0][col] = Fraction(1)  # b_{0,k0}
        V[1 * 14 + k1][col] = Fraction(1)  # b_{1,k1}
        # W: c_{0,k0} += m1_s -> W[k0*2+0][col] += 1
        # W: c_{1,k1} += m1_s -> W[k1*2+1][col] += 1
        W[k0 * 2 + 0][col] = Fraction(1)
        W[k1 * 2 + 1][col] = Fraction(1)
        
        # t=1: m2_s = (a10 + a11) * b_{0,k0}
        col = s * 7 + 1
        U[2][col] = Fraction(1)  # a10
        U[3][col] = Fraction(1)  # a11
        V[0 * 14 + k0][col] = Fraction(1)  # b_{0,k0}
        # W: c_{1,k0} += m2_s -> W[k0*2+1][col] += 1
        # W: c_{1,k1} -= m2_s -> W[k1*2+1][col] -= 1
        W[k0 * 2 + 1][col] = Fraction(1)
        W[k1 * 2 + 1][col] = Fraction(-1)
        
        # t=2: m3_s = a00 * (b_{0,k1} - b_{1,k1})
        col = s * 7 + 2
        U[0][col] = Fraction(1)  # a00
        V[0 * 14 + k1][col] = Fraction(1)  # b_{0,k1}
        V[1 * 14 + k1][col] = Fraction(-1)  # -b_{1,k1}
        # W: c_{0,k1} += m3_s -> W[k1*2+0][col] += 1
        # W: c_{1,k1} += m3_s -> W[k1*2+1][col] += 1
        W[k1 * 2 + 0][col] = Fraction(1)
        W[k1 * 2 + 1][col] = Fraction(1)
        
        # t=3: m4_s = a11 * (b_{1,k0} - b_{0,k0})
        col = s * 7 + 3
        U[3][col] = Fraction(1)  # a11
        V[1 * 14 + k0][col] = Fraction(1)  # b_{1,k0}
        V[0 * 14 + k0][col] = Fraction(-1)  # -b_{0,k0}
        # W: c_{0,k0} += m4_s -> W[k0*2+0][col] += 1
        # W: c_{1,k0} += m4_s -> W[k0*2+1][col] += 1
        W[k0 * 2 + 0][col] = Fraction(1)
        W[k0 * 2 + 1][col] = Fraction(1)
        
        # t=4: m5_s = (a00 + a01) * b_{1,k1}
        col = s * 7 + 4
        U[0][col] = Fraction(1)  # a00
        U[1][col] = Fraction(1)  # a01
        V[1 * 14 + k1][col] = Fraction(1)  # b_{1,k1}
        # W: c_{0,k0} -= m5_s -> W[k0*2+0][col] -= 1
        # W: c_{0,k1} += m5_s -> W[k1*2+0][col] += 1
        W[k0 * 2 + 0][col] = Fraction(-1)
        W[k1 * 2 + 0][col] = Fraction(1)
        
        # t=5: m6_s = (a10 - a00) * (b_{0,k0} + b_{0,k1})
        col = s * 7 + 5
        U[2][col] = Fraction(1)  # a10
        U[0][col] = Fraction(-1)  # -a00
        V[0 * 14 + k0][col] = Fraction(1)  # b_{0,k0}
        V[0 * 14 + k1][col] = Fraction(1)  # b_{0,k1}
        # W: c_{1,k1} += m6_s -> W[k1*2+1][col] += 1
        W[k1 * 2 + 1][col] = Fraction(1)
        
        # t=6: m7_s = (a01 - a11) * (b_{1,k0} + b_{1,k1})
        col = s * 7 + 6
        U[1][col] = Fraction(1)  # a01
        U[3][col] = Fraction(-1)  # -a11
        V[1 * 14 + k0][col] = Fraction(1)  # b_{1,k0}
        V[1 * 14 + k1][col] = Fraction(1)  # b_{1,k1}
        # W: c_{0,k0} += m7_s -> W[k0*2+0][col] += 1
        W[k0 * 2 + 0][col] = Fraction(1)
    
    # Convert to lists of floats (integers are fine)
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
