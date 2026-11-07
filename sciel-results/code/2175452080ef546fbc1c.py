"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure inputs are float64 arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    
    # The 'y' in the signature is the target variable for the evaluator's fit, 
    # but inside equation, we are predicting the target. 
    # Wait, the signature is equation(x, y, z, params). 
    # In the seed program: return params[0]*x + ...
    # The evaluator calls equation(x_train, y_train, z_train, params) to get predictions?
    # No, the contract says: "The evaluator fits params... minimizing the training MSE."
    # "equation must return an array with one prediction per point."
    # Usually, symbolic regression tasks provide x, y, z as inputs to the function, 
    # and the function returns the predicted value. The 'y' in the signature is likely 
    # the target values provided so the function *could* use them, but typically 
    # the function should only depend on x, z and params to make a prediction.
    # However, looking at the seed: return params[0]*x + params[1]*y + params[2]*z + params[3]
    # It uses 'y' as an input variable! 
    # This implies the input variables are named x, y, z in the function signature, 
    # but the target is NOT passed as 'y' if 'y' is used as a feature.
    # Let's re-read carefully: "Symbolic regression with 3 input(s)".
    # "Data: 200 training points... inputs drawn uniformly from [0.5, 2]... for x, y, z".
    # "targets are noiseless".
    # The signature is equation(x, y, z, params).
    # If the inputs are x, y, z, then the target is some function f(x,y,z).
    # The seed program uses x, y, z as features.
    # So, x, y, z are the independent variables. The target is hidden.
    
    # My strategy:
    # 1. Define a rich basis of functions of x, y, z.
    # 2. Use the provided 'params' as initial guess (all ones).
    # 3. Perform a local multi-start BFGS optimization to minimize MSE against the target.
    # BUT, I don't have the target! 
    # The evaluator fits the params. The function `equation` is called by the evaluator 
    # during the fitting process to compute the predictions.
    # The evaluator does: 
    #   def loss(p):
    #       pred = equation(x_train, y_train, z_train, p)
    #       return np.mean((pred - y_target_train)**2)
    #   res = minimize(loss, x0=np.ones(10), method='BFGS')
    #   pred_test = equation(x_test, y_test, z_test, res.x)
    #   
    # Therefore, `equation` is called MANY times during the fitting phase.
    # I CANNOT do expensive optimization inside `equation` because it will be called 
    # hundreds of times by the evaluator's `minimize` loop, and each call must be fast.
    # The 120s limit applies to the entire evaluation of one candidate program.
    # If I put a loop inside `equation`, it will run for every function evaluation 
    # in the BFGS optimizer, likely causing a timeout or extreme slowness.
    
    # CORRECTION: The task says "The evaluator fits params... starting from all ones".
    # This implies the evaluator calls `scipy.optimize.minimize` on a loss function 
    # that calls `equation`.
    # If I want to change the optimization strategy, I can't change the evaluator.
    # I can only change `equation`.
    # 
    # However, the prompt says: "You are exploring a DIFFERENT strategy... not refining 
    # the current program, but rethinking the approach."
    # And: "DIRECTIVE: COMPUTE BUDGET: the evaluator KILLS the program after 150s... 
    # Keep an EXPLICIT deadline guard... Spare time does not establish a bottleneck..."
    #
    # Is it possible to do the fitting INSIDE `equation`?
    # If `equation` is called with `params` that are just initial guesses, 
    # and I ignore them and do my own optimization using the data passed in...
    # But `equation` receives `x, y, z` which are the INPUTS. It does NOT receive the target values.
    # So I cannot compute the loss inside `equation` because I don't have the target `y_target`.
    # 
    # Wait, look at the signature again: `equation(x, y, z, params)`.
    # If `y` is an input variable (feature), I don't have the target.
    # If `y` is the target, then the inputs are only `x` and `z`? 
    # "Symbolic regression with 3 input(s)".
    # "inputs drawn uniformly from [0.5, 2]... for x, y, z".
    # This strongly implies x, y, z are the three independent variables.
    # So the target is not available inside `equation`.
    #
    # Therefore, I cannot fit the parameters inside `equation` because I lack the target.
    # I must rely on the evaluator's BFGS to fit the params.
    # 
    # So, how can I change the strategy?
    # 1. Change the basis function to be more expressive.
    # 2. Make the landscape smoother for BFGS to avoid local minima.
    # 3. Use a different parameterization that is easier to optimize.
    #
    # The previous attempts (p4, p5) used quadratic polynomials and scored ~7.4.
    # The linear model scored 1.0.
    # This suggests the true function is NOT well-approximated by a quadratic polynomial, 
    # OR the BFGS optimizer is getting stuck in a bad local minimum for the quadratic model, 
    # OR the linear model is actually the best fit among the candidates tried so far.
    #
    # If the linear model scores 1.0 (which is the baseline), and quadratics score 7.4 (worse), 
    # it's likely that BFGS is failing to find the correct quadratic coefficients. 
    # Why? Starting from all ones might be far from the optimum, and the quadratic 
    # landscape might have saddle points or sharp valleys.
    #
    # Strategy: Use a basis that is inherently smoother or more stable.
    # Or, better yet, use a model that is close to linear but slightly more flexible.
    #
    # Let's try a model with interaction terms but normalized inputs to help BFGS.
    # However, I cannot normalize inside `equation` because I don't know the mean/std 
    # of the test data, and normalizing changes the meaning of `params`.
    # But I can use fixed scaling based on the known domain [0.5, 2].
    # Centered and scaled inputs: x_c = (x - 1.25) / 0.75
    # This makes the inputs roughly in [-1, 1].
    # This often helps gradient-based optimizers.
    #
    # Let's try a centered linear model first to see if it helps BFGS.
    # If the linear model with centering scores better than 1.0, then the issue was scaling.
    #
    # Let's try a slightly richer model: Linear + some key interactions, centered.
    # Basis:
    # 1. 1
    # 2. x_c
    # 3. y_c
    # 4. z_c
    # 5. x_c * y_c
    # 6. x_c * z_c
    # 7. y_c * z_c
    # 8. x_c^2
    # 9. y_c^2
    # 10. z_c^2
    #
    # This is the same as the quadratic model but centered.
    # The previous quadratic model was uncentered.
    # Centering removes collinearity between constant and linear terms, 
    # and between linear and quadratic terms (partially).
    # This should make the Hessian better conditioned, helping BFGS.
    
    # Define centered variables
    x_c = (x - 1.25) / 0.75
    y_c = (y - 1.25) / 0.75
    z_c = (z - 1.25) / 0.75
    
    p = params
    
    # Construct the model
    # We have 10 params.
    pred = (p[0] * 1.0 +
            p[1] * x_c +
            p[2] * y_c +
            p[3] * z_c +
            p[4] * (x_c * y_c) +
            p[5] * (x_c * z_c) +
            p[6] * (y_c * z_c) +
            p[7] * (x_c * x_c) +
            p[8] * (y_c * y_c) +
            p[9] * (z_c * z_c))
            
    return pred
# EVOLVE-BLOCK-END
