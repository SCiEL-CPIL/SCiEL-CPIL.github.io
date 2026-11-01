"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # The optimal solution lies in the subspace spanned by H0 and H4
    # due to the monotonicity of the eigenvalues lambda_k = k + 0.5.
    # Constraint: c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=665280
    # Setting c8=0, c12=0 implies c0 + 12*c4 = 0 => c0 = -12*c4.
    # This yields the minimum Rayleigh quotient for the variance.
    return np.array([-12.0, 1.0, 0.0])
# EVOLVE-BLOCK-END
