import numpy as np
import time
import warnings

# Suppress warnings from numpy operations like division by zero or invalid values
warnings.filterwarnings("ignore")

# --- GP Node Representation ---
class Node:
    __slots__ = ['type', 'left', 'right', 'value']
    
    def __init__(self, type, left=None, right=None, value=None):
        self.type = type
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        if self.type == 'terminal':
            if self.value == 'x':
                return "x"
            else:
                return f"{self.value:.4f}"
        else:
            return f"({self.left} {self.type} {self.right})"

# --- Safe Mathematical Operations ---
def safe_eval(node, x):
    """
    Evaluates the node on array x.
    Returns array of floats.
    Handles NaN/Inf by returning large penalty values.
    """
    if node.type == 'terminal':
        if node.value == 'x':
            return x
        else:
            return np.full_like(x, node.value, dtype=float)
    
    # Get sub-expressions
    left_val = safe_eval(node.left, x)
    right_val = safe_eval(node.right, x)
    
    op = node.type
    result = None
    
    if op == '+':
        result = left_val + right_val
    elif op == '-':
        result = left_val - right_val
    elif op == '*':
        result = left_val * right_val
    elif op == '/':
        # Avoid division by zero
        mask = np.abs(right_val) > 1e-8
        result = np.zeros_like(left_val)
        result[mask] = left_val[mask] / right_val[mask]
        result[~mask] = 1e10 # Large penalty for division by zero
    elif op == 'sqrt':
        mask = left_val >= 0
        result = np.zeros_like(left_val)
        result[mask] = np.sqrt(left_val[mask])
        result[~mask] = 1e10
    elif op == 'log':
        mask = left_val > 0
        result = np.zeros_like(left_val)
        result[mask] = np.log(left_val[mask])
        result[~mask] = 1e10
    elif op == 'exp':
        # Clip to avoid overflow
        clipped = np.clip(left_val, -100, 100)
        result = np.exp(clipped)
    elif op == 'sin':
        result = np.sin(left_val)
    elif op == 'cos':
        result = np.cos(left_val)
    elif op == 'abs':
        result = np.abs(left_val)
    elif op == 'power':
        try:
            result = np.power(left_val, right_val)
            result[~np.isfinite(result)] = 1e10
        except:
            result = np.full_like(left_val, 1e10)
    elif op == 'max':
        result = np.maximum(left_val, right_val)
    elif op == 'min':
        result = np.minimum(left_val, right_val)
    else:
        result = left_val # Fallback
    
    # Final cleanup: replace any remaining non-finite with penalty
    result[~np.isfinite(result)] = 1e10
    return result

# --- GP Operators ---
TERMINALS = ['x', 'const']
OPERATORS = ['+', '-', '*', '/', 'sqrt', 'log', 'exp', 'sin', 'cos', 'abs', 'power', 'max', 'min']

def random_terminal():
    if np.random.random() < 0.5:
        return Node('terminal', value='x')
    else:
        return Node('terminal', value=float(np.random.uniform(-10, 10)))

def random_subtree(depth=0, max_depth=4):
    if depth >= max_depth:
        return random_terminal()
    
    if np.random.random() < 0.2: # Probability of stopping
        return random_terminal()
    
    op = np.random.choice(OPERATORS)
    left = random_subtree(depth + 1, max_depth)
    right = random_subtree(depth + 1, max_depth)
    return Node(op, left, right)

def get_nodes(node, nodes_list):
    nodes_list.append(node)
    if node.type != 'terminal':
        get_nodes(node.left, nodes_list)
        get_nodes(node.right, nodes_list)

def copy_tree(node):
    if node.type == 'terminal':
        return Node('terminal', value=node.value)
    return Node(node.type, copy_tree(node.left), copy_tree(node.right))

def mutate_tree(node, mutation_rate=0.1):
    if np.random.random() > mutation_rate:
        return node
    
    if node.type == 'terminal':
        if node.value == 'x':
            if np.random.random() < 0.5:
                return Node('terminal', value=float(np.random.uniform(-10, 10)))
            else:
                return Node('terminal', value='x')
        else:
            new_val = node.value + np.random.normal(0, 1)
            return Node('terminal', value=new_val)
    else:
        if np.random.random() < 0.5:
            new_op = np.random.choice(OPERATORS)
            left = mutate_tree(node.left, mutation_rate)
            right = mutate_tree(node.right, mutation_rate)
            return Node(new_op, left, right)
        else:
            left = mutate_tree(node.left, mutation_rate)
            right = mutate_tree(node.right, mutation_rate)
            return Node(node.type, left, right)

def evolve(X, y, time_limit=60.0, population_size=200, max_depth=5):
    start_time = time.time()
    
    # Initial Population
    population = []
    for _ in range(population_size):
        tree = random_subtree(0, max_depth)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    
    var_y = np.var(y)
    if var_y < 1e-10:
        var_y = 1.0
        
    def get_fitness(tree):
        try:
            pred = safe_eval(tree, X)
            mse = np.mean((pred - y)**2)
            nrmse = mse / var_y
            nodes = []
            get_nodes(tree, nodes)
            complexity = len(nodes)
            penalty = complexity * 0.01
            return nrmse + penalty
        except Exception:
            return 1e10

    fitnesses = [get_fitness(t) for t in population]
    
    gen = 0
    while time.time() - start_time < time_limit:
        gen += 1
        
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fitness:
            best_fitness = fitnesses[min_idx]
            best_tree = population[min_idx]
            
        if best_fitness < 1e-6:
            break
            
        new_population = []
        
        elites = sorted(range(population_size), key=lambda i: fitnesses[i])[:5]
        for i in elites:
            new_population.append(copy_tree(population[i]))
            new_population.append(copy_tree(population[i]))
            
        while len(new_population) < population_size:
            idx1 = np.random.randint(0, population_size)
            idx2 = np.random.randint(0, population_size)
            parent1 = population[idx1] if fitnesses[idx1] < fitnesses[idx2] else population[idx2]
            
            child = mutate_tree(copy_tree(parent1), mutation_rate=0.2)
            new_population.append(child)
            
        population = new_population
        fitnesses = [get_fitness(t) for t in population]

    if best_tree is None:
        best_tree = population[np.argmin(fitnesses)]
        
    return best_tree, best_fitness

# --- Fallback Models from Inspiration 1 ---

def create_poly_model(X_train, y_train, degree=12):
    """
    Creates a high-degree polynomial model with ridge regression.
    Returns a predict function.
    """
    X = X_train
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    
    X_norm = (X - x_min) / x_range
    
    def poly_features(X_n, d):
        cols = [np.ones(len(X_n))]
        for i in range(1, d + 1):
            cols.append(X_n[:, 0] ** i)
        return np.column_stack(cols)
    
    Phi_A = poly_features(X_norm, degree)
    lam_A = 1e-5
    try:
        w_A = np.linalg.solve(Phi_A.T @ Phi_A + lam_A * np.eye(degree + 1), Phi_A.T @ y_train)
        pred_A_train = Phi_A @ w_A
        mse_A = np.mean((pred_A_train - y_train) ** 2)
    except np.linalg.LinAlgError:
        w_A = np.zeros(degree + 1)
        pred_A_train = np.zeros_like(y_train)
        mse_A = np.inf

    def predict(X):
        X_n = (X - x_min) / x_range
        return poly_features(X_n, degree) @ w_A

    return predict, mse_A

def create_rbf_model(X_train, y_train, n_centers=25, gammas=None):
    """
    Creates an RBF kernel regression model.
    Returns a predict function and the best MSE.
    """
    if gammas is None:
        gammas = [1, 5, 10, 20, 50, 100, 200]
        
    X = X_train
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    
    X_norm = (X - x_min) / x_range
    
    centers = np.linspace(0, 1, n_centers).reshape(-1, 1)
    
    def rbf_features(X_n, gamma):
        diff = X_n - centers.T
        sq_dist = diff ** 2
        return np.exp(-gamma * sq_dist)
    
    best_mse_B = np.inf
    best_w_B = None
    best_gamma_B = None
    
    for gamma in gammas:
        K = rbf_features(X_norm, gamma)
        lam_B = 1e-4
        try:
            w_B = np.linalg.solve(K.T @ K + lam_B * np.eye(n_centers), K.T @ y_train)
            pred_B_train = K @ w_B
            mse_B = np.mean((pred_B_train - y_train) ** 2)
            
            if mse_B < best_mse_B:
                best_mse_B = mse_B
                best_w_B = w_B
                best_gamma_B = gamma
        except np.linalg.LinAlgError:
            continue

    if best_w_B is None:
        return None, np.inf

    def predict(X):
        X_n = (X - x_min) / x_range
        return rbf_features(X_n, best_gamma_B) @ best_w_B

    return predict, best_mse_B

def fit(X_train, y_train):
    start_time = time.time()
    # Total limit is 330s. We will allocate ~290s for safety.
    # We split the time: 60% for GP, 40% for Fallbacks (or just run fallbacks first to get a baseline quickly?)
    # Actually, GP is slow. Let's run GP first with a generous budget, but keep a hard deadline.
    # Or better: Run GP for a fixed time, then if time remains, try to refine or just use the best found.
    # The inspiration 1 models are very fast. Let's compute them first as a "safety net".
    
    # 1. Compute Fallback Models (Fast)
    t0 = time.time()
    poly_pred_func, mse_poly = create_poly_model(X_train, y_train, degree=12)
    rbf_pred_func, mse_rbf = create_rbf_model(X_train, y_train)
    
    # Determine the best fallback
    best_fallback_func = None
    best_fallback_mse = np.inf
    
    if mse_poly < best_fallback_mse:
        best_fallback_func = poly_pred_func
        best_fallback_mse = mse_poly
        
    if mse_rbf < best_fallback_mse:
        best_fallback_func = rbf_pred_func
        best_fallback_mse = mse_rbf
        
    # If RBF is significantly better, use it, otherwise Poly (as per inspiration 1 logic)
    # But for now, just pick the one with lower training MSE.
    
    # 2. Run GP Evolution
    # Allocate remaining time to GP.
    # We want to ensure GP finishes before the 330s limit.
    # Let's give GP 200 seconds.
    gp_time_limit = 200.0
    
    # If we have already used a lot of time, reduce GP time.
    elapsed = time.time() - start_time
    if elapsed > 10:
        gp_time_limit = max(10, 200 - elapsed)
        
    best_gp_tree = None
    best_gp_fitness = np.inf
    
    try:
        best_gp_tree, best_gp_fitness = evolve(X_train, y_train, time_limit=gp_time_limit, population_size=200, max_depth=5)
    except Exception:
        pass
    
    # 3. Select the best model
    # Compare the training performance of the GP model vs the fallbacks.
    # We need to calculate the normalized MSE for the GP model on training data to compare fairly.
    
    var_y = np.var(y_train)
    if var_y < 1e-10:
        var_y = 1.0
        
    gp_norm_mse = np.inf
    if best_gp_tree is not None:
        try:
            pred_gp = safe_eval(best_gp_tree, X_train[:, 0])
            mse_gp = np.mean((pred_gp - y_train)**2)
            gp_norm_mse = mse_gp / var_y
            # Add parsimony penalty for comparison? The fallbacks don't have one.
            # Let's just compare raw normalized MSE.
        except Exception:
            gp_norm_mse = np.inf
    
    # Fallback normalized MSEs
    poly_norm_mse = mse_poly / var_y
    rbf_norm_mse = mse_rbf / var_y
    
    # Choose the model with the lowest normalized training MSE
    # If GP is not significantly better, the fallback might be more stable on test data.
    # However, symbolic regression *should* generalize better if it finds the exact formula.
    # If GP finds a very low error (< 1e-4), it likely found the formula.
    
    best_model_func = None
    best_score = np.inf
    
    # Candidates:
    # 1. GP
    if gp_norm_mse < best_score:
        best_score = gp_norm_mse
        best_model_func = lambda X: safe_eval(best_gp_tree, X[:, 0])
        
    # 2. Poly
    if poly_norm_mse < best_score:
        best_score = poly_norm_mse
        best_model_func = poly_pred_func
        
    # 3. RBF
    if rbf_norm_mse < best_score:
        best_score = rbf_norm_mse
        best_model_func = rbf_pred_func
        
    # Final safety check
    if best_model_func is None:
        # Absolute fallback: mean
        mean_y = np.mean(y_train)
        best_model_func = lambda X: np.full(X.shape[0], mean_y)
        
    # Wrap the final prediction function to ensure correct shape and safety
    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        try:
            pred = best_model_func(X)
            pred = np.asarray(pred, dtype=float)
            # Ensure 1D output
            if pred.ndim > 1:
                pred = pred.flatten()
            # Handle any non-finite values
            pred[~np.isfinite(pred)] = np.mean(y_train)
            return pred
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
            
    return predict

