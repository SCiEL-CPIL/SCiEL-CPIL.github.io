import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Map inputs from [0.5, 2] to [-1, 1] for Chebyshev polynomials
    # t = (2*var - (0.5 + 2)) / (2 - 0.5) = (2*var - 2.5) / 1.5
    tx = (2.0 * x - 2.5) / 1.5
    ty = (2.0 * y - 2.5) / 1.5
    tz = (2.0 * z - 2.5) / 1.5
    
    # Compute Chebyshev polynomials up to degree 3 for each variable
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    T0_x = np.ones_like(tx)
    T1_x = tx
    T2_x = 2.0 * tx**2 - 1.0
    T3_x = 4.0 * tx**3 - 3.0 * tx
    
    T0_y = np.ones_like(ty)
    T1_y = ty
    T2_y = 2.0 * ty**2 - 1.0
    T3_y = 4.0 * ty**3 - 3.0 * ty
    
    T0_z = np.ones_like(tz)
    T1_z = tz
    T2_z = 2.0 * tz**2 - 1.0
    T3_z = 4.0 * tz**3 - 3.0 * tz
    
    # Basis selection with 10 parameters:
    # We use a mix of univariate terms up to degree 3 and key interactions.
    # The complete quadratic polynomial basis (1, x, y, z, x^2, y^2, z^2, xy, xz, yz)
    # spans the same space as {1, T1, T2, T1*T1} terms.
    # To go beyond quadratic, we include cubic univariate terms (T3) and keep some interactions.
    
    # Let's try a basis that includes:
    # 0: 1 (constant)
    # 1: T1_x (linear x)
    # 2: T1_y (linear y)
    # 3: T1_z (linear z)
    # 4: T2_x (quadratic x)
    # 5: T2_y (quadratic y)
    # 6: T2_z (quadratic z)
    # 7: T3_x (cubic x)
    # 8: T3_y (cubic y)
    # 9: T3_z (cubic z)
    
    # This is a univariate basis up to degree 3. It might miss interactions.
    # Alternative:
    # 0: 1
    # 1: T1_x
    # 2: T1_y
    # 3: T1_z
    # 4: T2_x
    # 5: T2_y
    # 6: T2_z
    # 7: T1_x * T1_y (interaction xy)
    # 8: T1_x * T1_z (interaction xz)
    # 9: T1_y * T1_z (interaction yz)
    # This is the complete quadratic basis. Score was ~7.39.
    
    # To improve, we need to capture non-polynomial or higher-order behavior.
    # Since the directive says "KEEP this design" (Chebyshev basis) and "investigate improvements within its family",
    # and the previous best score with this family was 7.39, we should try to refine it.
    
    # One idea: Use a rational function form with Chebyshev polynomials?
    # Or use a different combination of terms.
    
    # Let's try a basis that includes some cubic terms AND interactions.
    # But we only have 10 params.
    
    # Option A: Univariate up to degree 4?
    # T4(t) = 8t^4 - 8t^2 + 1
    # Basis: 1, T1_x, T1_y, T1_z, T2_x, T2_y, T2_z, T3_x, T3_y, T3_z (10 terms)
    # This allows for more complex univariate shapes.
    
    # Option B: Quadratic + one cubic interaction?
    # 1, T1_x, T1_y, T1_z, T2_x, T2_y, T2_z, T1_x*T1_y, T1_x*T1_z, T1_y*T1_z (10 terms)
    # This is the standard quadratic basis.
    
    # Option C: A mix.
    # Let's try Option A first, as it explores the "higher degree" direction within the Chebyshev family.
    
    T4_x = 8.0 * tx**4 - 8.0 * tx**2 + 1.0
    T4_y = 8.0 * ty**4 - 8.0 * ty**2 + 1.0
    T4_z = 8.0 * tz**4 - 8.0 * tz**2 + 1.0
    
    # Let's try a basis that includes T3 and T4 for at least one variable, or a mix.
    # Actually, the most general form with 10 params using Chebyshev is just a linear combination.
    # The key is which basis functions to choose.
    
    # Let's stick to the complete quadratic basis (Option B) as it was the most robust,
    # but maybe we can add a small non-linear correction?
    
    # Wait, the directive says "Implement a basis of Chebyshev polynomials... to leverage their minimax property".
    # The previous program (score 32700491930) was a power law.
    # The inspiration (score 13546) was rational.
    # The best observed score in the JSON for a Chebyshev-like program was 7.39.
    
    # Let's implement the complete quadratic basis in Chebyshev coordinates, which is equivalent to the monomial quadratic basis
    # but potentially better conditioned.
    
    basis = np.column_stack([
        T0_x,          # 1
        T1_x,          # x
        T1_y,          # y
        T1_z,          # z
        T2_x,          # x^2
        T2_y,          # y^2
        T2_z,          # z^2
        T1_x * T1_y,   # xy
        T1_x * T1_z,   # xz
        T1_y * T1_z    # yz
    ])
    
    result = basis @ p
    
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        
    return result
