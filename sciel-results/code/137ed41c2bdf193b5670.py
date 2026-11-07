"""Symbolic regression using Genetic Programming (Evolutionary Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator

# Define the GP operators
# Terminals: x0, x1, ..., x_{d-1}
# Functions: +, -, *, /, sin, cos, tan, exp, log, abs, sqrt, pow(x, 2), pow(x, 3)
# Note: We use safe versions of functions to handle domain issues gracefully during evolution.

def safe_div(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = a / b
    # If b is close to 0, the result is huge. We cap it to avoid gradient explosion in fitness
    # but for pure symbolic regression, we just need finite values.
    # Actually, for GP fitness, we want to penalize instability.
    # Let's just return 0 where b is 0 to keep it simple and finite, or a large constant.
    # Better: return 0 where b is 0.
    res[np.abs(b) < 1e-8] = 0.0
    return res

def safe_log(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.log(np.abs(a) + 1e-8)
    return res

def safe_exp(a):
    # Clip to avoid overflow
    a = np.clip(a, -50, 50)
    return np.exp(a)

def safe_tan(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.tan(a)
    # Tan has singularities. Cap it.
    res[np.abs(res) > 100] = np.sign(res[np.abs(res) > 100]) * 100
    return res

def safe_sqrt(a):
    return np.sqrt(np.abs(a))

def safe_pow2(a):
    return a * a

def safe_pow3(a):
    return a * a * a

class GP:
    def __init__(self, n_features, max_depth=4):
        self.n_features = n_features
        self.max_depth = max_depth
        self.functions = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': safe_div,
            'sin': np.sin,
            'cos': np.cos,
            'tan': safe_tan,
            'exp': safe_exp,
            'log': safe_log,
            'abs': np.abs,
            'sqrt': safe_sqrt,
            'pow2': safe_pow2,
            'pow3': safe_pow3
        }
        self.terminals = [f'x{i}' for i in range(n_features)]
        
    def create_tree(self, depth=0):
        """Create a random tree using grow method."""
        if depth >= self.max_depth or (depth > 0 and random.random() < 0.1):
            return random.choice(self.terminals)
        
        func = random.choice(list(self.functions.keys()))
        if func in ['+', '-', '*', '/', 'sin', 'cos', 'tan', 'exp', 'log', 'abs', 'sqrt', 'pow2', 'pow3']:
            # All our functions are unary or binary. 
            # Let's simplify: assume all are binary for ease, or handle unary.
            # My dict has unary and binary mixed. 
            # Let's define arity explicitly.
            pass
        
        # Let's refactor to explicit arity
        if random.random() < 0.3: # 30% chance of terminal if depth allows
             return random.choice(self.terminals)
             
        # Choose a function
        func_name = random.choice(list(self.functions.keys()))
        # Determine arity
        if func_name in ['+', '-', '*', '/']:
            left = self.create_tree(depth + 1)
            right = self.create_tree(depth + 1)
            return (func_name, left, right)
        else:
            child = self.create_tree(depth + 1)
            return (func_name, child)
            
    def evaluate_tree(self, tree, X):
        """Recursively evaluate the tree for all samples in X."""
        if isinstance(tree, str):
            # It's a terminal
            idx = int(tree[1:])
            return X[:, idx]
        
        func_name = tree[0]
        func = self.functions[func_name]
        
        if func_name in ['+', '-', '*', '/']:
            left_val = self.evaluate_tree(tree[1], X)
            right_val = self.evaluate_tree(tree[2], X)
            return func(left_val, right_val)
        else:
            child_val = self.evaluate_tree(tree[1], X)
            return func(child_val)

    def crossover(self, tree1, tree2):
        """Standard GP crossover."""
        # Pick a random node from tree1
        nodes1 = self.get_nodes(tree1)
        nodes2 = self.get_nodes(tree2)
        
        if not nodes1 or not nodes2:
            return tree1, tree2
            
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        
        # Swap
        new_tree1 = self.replace_node(tree1, node1, node2)
        new_tree2 = self.replace_node(tree2, node2, node1)
        
        return new_tree1, new_tree2

    def get_nodes(self, tree):
        """Get all nodes in the tree."""
        nodes = []
        if isinstance(tree, str):
            nodes.append(tree)
        else:
            nodes.append(tree)
            if tree[0] in ['+', '-', '*', '/']:
                nodes.extend(self.get_nodes(tree[1]))
                nodes.extend(self.get_nodes(tree[2]))
            else:
                nodes.extend(self.get_nodes(tree[1]))
        return nodes

    def replace_node(self, tree, target, replacement):
        """Replace the first occurrence of target with replacement."""
        if isinstance(tree, str):
            if tree == target:
                return replacement
            return tree
        
        if tree == target:
            return replacement
            
        func_name = tree[0]
        if func_name in ['+', '-', '*', '/']:
            new_left = self.replace_node(tree[1], target, replacement)
            new_right = self.replace_node(tree[2], target, replacement)
            return (func_name, new_left, new_right)
        else:
            new_child = self.replace_node(tree[1], target, replacement)
            return (func_name, new_child)

    def mutate(self, tree, mutation_rate=0.1):
        """Standard GP mutation."""
        nodes = self.get_nodes(tree)
        if not nodes:
            return tree
            
        # Pick a random node to mutate
        target = random.choice(nodes)
        
        if random.random() < 0.5:
            # Replace with a random terminal
            replacement = random.choice(self.terminals)
        else:
            # Replace with a random small subtree
            replacement = self.create_tree(depth=1)
            
        return self.replace_node(tree, target, replacement)

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285.0  # 15s safety margin
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n_samples, n_features = X.shape
    
    # Normalization for stability in GP
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-12] = 1.0
    X_norm = (X - X_min) / X_range
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    y_norm = (y - y_mean) / y_std
    
    gp = GP(n_features, max_depth=5)
    
    # Initialize population
    pop_size = 200
    population = [gp.create_tree(depth=2) for _ in range(pop_size)]
    
    best_fitness = -np.inf
    best_tree = None
    best_linear_weights = None
    best_linear_offset = None
    
    def evaluate_fitness(tree):
        """
        Evaluate tree. 
        Strategy: 
        1. Generate features from tree.
        2. Fit a linear regression y = w * f(x) + b.
        3. Return R^2 or negative MSE.
        This allows the GP to find the *form* of the function, while linear regression handles the scaling.
        """
        try:
            f_vals = gp.evaluate_tree(tree, X_norm)
            if not np.all(np.isfinite(f_vals)):
                return -np.inf
            
            # Add intercept
            F = np.column_stack([f_vals, np.ones(n_samples)])
            
            # Ridge regression for stability
            # We want to find w such that F @ w ~ y_norm
            try:
                w, *_ = np.linalg.lstsq(F, y_norm, rcond=None)
                y_pred_norm = F @ w
                mse = np.mean((y_pred_norm - y_norm) ** 2)
                # Higher fitness is better
                fitness = -mse
                return fitness, w
            except:
                return -np.inf, None
        except Exception:
            return -np.inf, None

    # Evaluate initial population
    for i, tree in enumerate(population):
        res = evaluate_fitness(tree)
        if isinstance(res, tuple):
            fitness, w = res
        else:
            fitness, w = res, None
            
        if fitness > best_fitness:
            best_fitness = fitness
            best_tree = tree
            best_linear_weights = w

    # Evolution loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate all individuals
        fitnesses = []
        weights_list = []
        for tree in population:
            res = evaluate_fitness(tree)
            if isinstance(res, tuple):
                f, w = res
            else:
                f, w = res, None
            fitnesses.append(f)
            weights_list.append(w)
            
            if f > best_fitness:
                best_fitness = f
                best_tree = tree
                best_linear_weights = w

        # Sort by fitness
        sorted_indices = np.argsort(fitnesses)[::-1]
        new_population = []
        
        # Elitism: keep top 10%
        elite_size = max(1, pop_size // 10)
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            t1_idx = random.randint(0, pop_size - 1)
            t2_idx = random.randint(0, pop_size - 1)
            parent1 = population[t1_idx]
            parent2 = population[t2_idx]
            
            # Crossover
            if random.random() < 0.7:
                child1, child2 = gp.crossover(parent1, parent2)
            else:
                child1 = parent1
                child2 = parent2
            
            # Mutation
            child1 = gp.mutate(child1, mutation_rate=0.2)
            child2 = gp.mutate(child2, mutation_rate=0.2)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
                
        population = new_population[:pop_size]
        
        # Check if we should break early if fitness is very good
        if best_fitness > -1e-6:
             break

    # Final check: ensure we have a valid model
    if best_tree is None:
        # Fallback to linear
        F = np.column_stack([X_norm, np.ones(n_samples)])
        w, *_ = np.linalg.lstsq(F, y_norm, rcond=None)
        def predict(X):
            Xn = (X - X_min) / X_range
            F = np.column_stack([Xn, np.ones(len(Xn))])
            return (F @ w) * y_std + y_mean
        return predict

    # Construct final predictor
    # The best tree produces a feature f(x). We fit y = w0 * f(x) + w1
    # w in best_linear_weights is [w_feature, w_intercept]
    
    def predict(X):
        try:
            Xn = (X - X_min) / X_range
            f_vals = gp.evaluate_tree(best_tree, Xn)
            if not np.all(np.isfinite(f_vals)):
                f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply linear weights
            # y_norm = w[0] * f + w[1]
            y_pred_norm = best_linear_weights[0] * f_vals + best_linear_weights[1]
            y_pred = y_pred_norm * y_std + y_mean
            
            # Safety: if any non-finite, replace with mean
            if not np.all(np.isfinite(y_pred)):
                y_pred = np.nan_to_num(y_pred, nan=y_mean, posinf=y_mean, neginf=y_mean)
                
            return y_pred
        except Exception:
            return np.full(len(X), y_mean)

    return predict

# EVOLVE-BLOCK-END
