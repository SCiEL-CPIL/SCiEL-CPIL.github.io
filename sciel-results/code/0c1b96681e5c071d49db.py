"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Import-Time Basis Search (ITBS) with Linear Pre-fitting.
    
    The current program uses a fixed set of 10 global monomials/trig functions.
    This approach is structurally rigid. If the true dynamics involve specific 
    local nonlinearities or different harmonic components, the fixed basis may 
    be suboptimal.
    
    New Strategy:
    1. Define a larger library of candidate basis functions (e.g., 30-40 terms) 
       including:
       - Constants
       - Time harmonics: sin(k*t), cos(k*t) for k=1..5
       - State polynomials: x, v, x^2, v^2, x^3, v^3, x*v, x^2*v, v^2*x
       - Local RBFs in (x,v) space with a few fixed centers (e.g., 0,0; 1,1; -1,-1)
       - Local RBFs in t space (less likely but cheap to test)
       
    2. At import time (when `equation` is defined or first called? No, must be 
       inside equation or module level? The prompt says "The evaluator fits params... 
       for each candidate". It does NOT say we can run code at import time that 
       sees the data. 
       
       WAIT. The prompt says: "You do not see the data, only the score of each 
       candidate." and "The evaluator fits params... starting from all ones".
       
       CRITICAL CONSTRAINT CHECK: 
       "The evaluator measures the raw value... The search score is seed/value..."
       "You do not see the data".
       
       Therefore, I CANNOT run `np.linalg.lstsq` on the training data inside the 
       `equation` function if I don't have access to the training target `y`. 
       The `equation` function signature is `equation(t, x, v, params)`. It does 
       NOT receive `y`. 
       
       Thus, I cannot do an "Import-Time Basis Search" that minimizes MSE on the 
       training data because I don't have `y` inside the function. I can only 
       compute the features.
       
       RE-EVALUATION OF STRATEGY:
       The "Import-Time Basis Search" in Inspiration 1 (score 202) likely refers 
       to selecting a basis that is *structurally* likely to be well-conditioned 
       or covers the space better, OR it implies that the *search* happens in the 
       outer loop (SCiEL) by proposing different hardcoded bases.
       
       However, the directive says: "Implement a Neural ODE-inspired Residual Basis...".
       
       Let's look at the Inspiration 1 again: "score 202.9388... uses numpy, time; 
       defines _make_basis, _build_surrogate, _fit_linear, _select_best_basis, equation".
       This implies the *program* contains logic to select a basis. But how? 
       If I can't see `y`, I can't select the best basis for the data.
       
       UNLESS: The "score" is from a previous run where the *outer loop* (SCiEL) 
       mutated the code to try different bases. The "Inspiration" is just a 
       successful *variant* of the code, not necessarily a dynamic runtime search 
       that I can replicate without `y`.
       
       Wait, look at the current program's score: 10474.9729.
       Inspiration 1: 202.9388.
       Inspiration 3: 4109.5809.
       
       The current program is BAD. It uses a fixed global basis.
       
       What is a "MATERIALLY DIFFERENT" strategy?
       The directive suggests: "Neural ODE-inspired Residual Basis using a single 
       hidden layer with a fixed, non-learnable weight matrix and a tanh activation 
       function, combined with a linear skip connection".
       
       This is a fixed architecture, but different from the current monomial basis.
       It uses `tanh` non-linearity which is bounded and smooth, potentially 
       better conditioned than high-order polynomials (x^5, v^3) which can 
       explode or be ill-conditioned if the input ranges are not perfectly scaled 
       or if the true function has sharp features.
       
       Let's implement the Neural ODE-inspired Residual Basis as directed.
       
       Architecture:
       Inputs: t, x, v (3 features)
       Hidden Layer: 
         H = tanh(W_fixed @ [t, x, v] + b_fixed)
       We need to map this to 10 parameters.
       
       The `params` array has length 10.
       We can split the 10 params:
       - 3 params for the linear skip connection: w_t, w_x, w_v
       - 1 param for bias
       - 6 params for the hidden layer output combination?
       
       Or, more simply:
       Create a fixed basis of 10 functions that are outputs of a small "neural" 
       network with fixed random weights, then the `params` are the linear 
       coefficients for these 10 basis functions.
       
       Let's define 10 basis functions:
       1. 1 (bias)
       2. t
       3. x
       4. v
       5. tanh(t)
       6. tanh(x)
       7. tanh(v)
       8. tanh(x*v)
       9. tanh(x + v)
       10. tanh(t + x)
       
       This is a "Neural-inspired" basis. It uses bounded non-linearities.
       
       Alternative: 
       The directive says: "fixed, non-learnable weight matrix and a tanh activation 
       function, combined with a linear skip connection".
       
       Let's construct the features explicitly using fixed weights to create 
       diverse non-linear combinations.
       
       Features:
       f1 = 1
       f2 = t
       f3 = x
       f4 = v
       f5 = tanh(1.0*t + 0.5*x)
       f6 = tanh(0.5*t + 1.0*v)
       f7 = tanh(1.0*x + 0.5*v)
       f8 = tanh(0.5*x + 1.0*v + 0.5*t)
       f9 = tanh(x*v)
       f10 = tanh(x**2 + v**2)
       
       This creates a set of 10 non-linear, bounded features. The BFGS optimizer 
       will fit the linear combination of these. This avoids the ill-conditioning 
       of high-degree polynomials (like x^5) and the locality issues of RBFs (if 
       centers are poorly chosen). The tanh functions provide smooth, global 
       non-linearities that can approximate a wide range of functions.
       
       Let's refine the fixed weights to ensure diversity.
       """
    
    t = np.asarray(t)
    x = np.asarray(x)
    v = np.asarray(v)
    
    # Ensure shapes are consistent
    if t.ndim == 0:
        t = t.reshape(1)
    if x.ndim == 0:
        x = x.reshape(1)
    if v.ndim == 0:
        v = v.reshape(1)
        
    # Basis Functions
    # 1. Bias
    b1 = np.ones_like(t)
    
    # 2-4. Linear inputs (Skip connection)
    b2 = t
    b3 = x
    b4 = v
    
    # 5-10. Non-linear features using tanh with fixed "weights"
    # These are designed to capture interactions and non-linearities 
    # without the risk of polynomial explosion.
    
    # Interaction in state space
    b5 = np.tanh(x * v)
    
    # Non-linear stiffness/damping approximations
    b6 = np.tanh(x)
    b7 = np.tanh(v)
    
    # Mixed time-state interactions
    b8 = np.tanh(t * x)
    b9 = np.tanh(t * v)
    
    # Quadratic-like non-linearity bounded by tanh
    b10 = np.tanh(x**2 + v**2)
    
    features = np.column_stack((
        b1, b2, b3, b4, b5, b6, b7, b8, b9, b10
    ))
    
    return features @ params

# EVOLVE-BLOCK-END
