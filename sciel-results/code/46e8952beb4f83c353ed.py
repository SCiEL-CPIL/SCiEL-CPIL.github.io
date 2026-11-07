"""
Genetic Programming Symbolic Regression for 1-D Noiseless Equations.

Strategy:
1. Represent mathematical expressions as trees (GPNode).
2. Initialize a population of random trees.
3. Evolve the population using tournament selection, crossover, and mutation.
4. Fitness function:
   - Primary: Mean Squared Error (MSE) on training data.
   - Secondary: Complexity penalty (number of nodes) to encourage simpler expressions.
   - Since data is noiseless, we aim for near-zero MSE with minimal complexity.
5. Return the best found expression as a predictor.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
import math

# Deadline guard: 330s wall-clock limit, use 290s for safety
_DEADLINE = time.time() + 290.0

# Operators and Functions
# We define safe versions to handle edge cases (div by zero, log of negative, etc.)
def _safe_div(a, b):
    return np.where(np.abs(b) > 1e-12, a / b, 0.0)

def _safe_log(a):
    return np.log(np.abs(a) + 1e-12)

def _safe_sqrt(a):
    return np.sqrt(np.abs(a))

def _safe_exp(a):
    # Clip to prevent overflow
    a_clipped = np.clip(a, -50, 50)
    return np.exp(a_clipped)

def _safe_sin(a):
    return np.sin(a)

def _safe_cos(a):
    return np.cos(a)

def _safe_tan(a):
    # Tan has singularities, clip result
    res = np.tan(a)
    return np.clip(res, -1e6, 1e6)

def _safe_abs(a):
    return np.abs(a)

def _safe_pow(a, b):
    # Only allow integer powers or small fractional? 
    # For GP, standard pow is risky. Let's restrict to simple powers if possible or just use * and +
    # Actually, for symbolic regression of smooth functions, x^2, x^3 are common.
    # Let's use a safe pow that handles negative bases with integer exponents.
    # If b is not integer, we might have issues. 
    # Given the hidden equations are likely standard math functions, 
    # we'll include +, -, *, /, sin, cos, exp, log, sqrt, abs, and constants.
    # We will exclude generic pow to avoid complexity explosion and numerical issues, 
    # relying on * for x*x.
    pass 

# Terminals: x, constants
# Functions: +, -, *, /, sin, cos, exp, log, sqrt, abs

class GPNode:
    def __init__(self, value=None, func=None, left=None, right=None):
        self.value = value  # If terminal
        self.func = func    # If internal
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.func is None:
            # Terminal
            if self.value == 'x':
                return x
            else:
                return np.full_like(x, self.value, dtype=float)
        else:
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)
            return self.func(left_val, right_val)

    def get_size(self):
        if self.func is None:
            return 1
        else:
            return 1 + self.left.get_size() + self.right.get_size()

    def get_depth(self):
        if self.func is None:
            return 0
        else:
            return 1 + max(self.left.get_depth(), self.right.get_depth())

    def __str__(self):
        if self.func is None:
            return str(self.value)
        else:
            return f"({self.func.__name__}({self.left}, {self.right}))"

# Define functions
FUNCTIONS = [
    operator.add,
    operator.sub,
    operator.mul,
    _safe_div,
    _safe_sin,
    _safe_cos,
    _safe_exp,
    _safe_log,
    _safe_sqrt,
    _safe_abs
]

TERMINALS = ['x', 1.0, 2.0, 3.0, 0.5, -1.0]

def _create_random_tree(max_depth=5):
    if max_depth == 0 or (random.random() < 0.1 and max_depth > 0):
        # Terminal
        term = random.choice(TERMINALS)
        return GPNode(value=term)
    else:
        func = random.choice(FUNCTIONS)
        left = _create_random_tree(max_depth - 1)
        right = _create_random_tree(max_depth - 1)
        return GPNode(func=func, left=left, right=right)

def _get_subtree(node):
    # Returns a random subtree
    if node.func is None:
        return node
    choice = random.random()
    if choice < 0.33:
        return node
    elif choice < 0.66:
        return _get_subtree(node.left)
    else:
        return _get_subtree(node.right)

def _crossover(node1, node2):
    # Subtree crossover
    # Get a random subtree from node1
    cut_point1 = _get_subtree(node1)
    # Get a random subtree from node2
    cut_point2 = _get_subtree(node2)
    
    # Swap
    parent1 = None
    parent2 = None
    
    # This simple approach creates a new tree structure by copying and swapping.
    # To be efficient, we'd do in-place, but for n=200 and small trees, copying is fine.
    
    def copy_tree(node):
        if node.func is None:
            return GPNode(value=node.value)
        return GPNode(func=node.func, left=copy_tree(node.left), right=copy_tree(node.right))
    
    def swap(node, target, replacement):
        if node is target:
            return replacement
        if node.func is None:
            return node
        new_left = swap(node.left, target, replacement)
        new_right = swap(node.right, target, replacement)
        return GPNode(func=node.func, left=new_left, right=new_right)
    
    # Find the parent of cut_point1 in node1? 
    # Easier: Just create new trees from the cut points?
    # No, standard GP crossover:
    # 1. Pick a point in T1.
    # 2. Pick a point in T2.
    # 3. Swap the subtrees rooted at those points.
    
    # Since we don't keep parent pointers, we do a recursive swap.
    # We need to identify the node object to swap.
    
    new_t1 = swap(node1, cut_point1, cut_point2)
    new_t2 = swap(node2, cut_point2, cut_point1)
    
    return new_t1, new_t2

def _mutate(node, mutation_rate=0.1):
    if node.func is None:
        # Change terminal
        if random.random() < mutation_rate:
            return GPNode(value=random.choice(TERMINALS))
        return node
    
    # Change function
    if random.random() < mutation_rate:
        new_func = random.choice(FUNCTIONS)
        new_left = _mutate(node.left, mutation_rate)
        new_right = _mutate(node.right, mutation_rate)
        return GPNode(func=new_func, left=new_left, right=new_right)
    
    # Mutate children
    new_left = _mutate(node.left, mutation_rate)
    new_right = _mutate(node.right, mutation_rate)
    return GPNode(func=node.func, left=new_left, right=new_right)

def _fit_gp(x_train, y_train, time_budget=280.0):
    start_time = time.time()
    
    # Normalize x to [-1, 1] or [0, 1] to help GP? 
    # The problem gives ranges like [-3, 3]. 
    # GP with raw x might struggle with scaling. 
    # Let's normalize x for the GP search, but the final predictor must use raw x.
    # If we evolve an expression in terms of normalized x, we can just substitute 
    # (x - min) / (max - min) into the expression? 
    # Or we can just let GP use raw x. 
    # Raw x is safer for "symbolic" interpretation, but scaling helps convergence.
    # Let's try raw x first, as the functions (sin, cos) are periodic and scale invariant in a way, 
    # but polynomials are not. 
    # Actually, let's normalize x to [0, 1] for the GP search space, 
    # and then the returned function will implicitly use the normalized version?
    # No, the predict function receives raw X. 
    # If we evolve f(x_norm), the predictor must compute x_norm = (x - x_min) / (x_max - x_min) 
    # and then evaluate f(x_norm).
    # This is a valid strategy.
    
    x_min = np.min(x_train)
    x_max = np.max(x_train)
    if x_max == x_min:
        x_norm = np.zeros_like(x_train)
    else:
        x_norm = (x_train - x_min) / (x_max - x_min)
    
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-9:
        y_norm = np.zeros_like(y_train)
    else:
        y_norm = (y_train - y_mean) / y_std
    
    # Initialize population
    pop_size = 200
    population = []
    for _ in range(pop_size):
        tree = _create_random_tree(max_depth=4)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    
    # Evaluate initial population
    for tree in population:
        try:
            pred = tree.evaluate(x_norm)
            mse = np.mean((pred - y_norm) ** 2)
            complexity = tree.get_size()
            # Fitness: MSE + lambda * complexity
            # Since data is noiseless, we want MSE ~ 0.
            # Penalty for complexity to avoid overfitting to noise (if any) or trivial solutions.
            fitness = mse + 0.01 * complexity
            if fitness < best_fitness:
                best_fitness = fitness
                best_tree = tree
        except Exception:
            continue
            
    # Evolution loop
    generation = 0
    max_generations = 10000
    while time.time() < start_time + time_budget and generation < max_generations:
        generation += 1
        
        # Selection: Tournament
        def select_tournament(pop, k=3):
            candidates = random.sample(pop, k)
            best_candidate = None
            best_fit = np.inf
            for c in candidates:
                # We need to store fitness with the tree? 
                # For simplicity, let's store (tree, fitness) in population
                pass
            
        # Refactor: Population is list of (tree, fitness)
        
        # Let's restart the loop logic with proper structure
        break # The previous loop was a sketch. Let's implement properly below.

    # Proper GP Implementation
    population = []
    for _ in range(pop_size):
        tree = _create_random_tree(max_depth=4)
        population.append(tree)
        
    # Calculate fitness for all
    def calc_fitness(tree):
        try:
            pred = tree.evaluate(x_norm)
            mse = np.mean((pred - y_norm) ** 2)
            complexity = tree.get_size()
            return mse + 0.01 * complexity
        except Exception:
            return np.inf

    # Initial evaluation
    for i, tree in enumerate(population):
        population[i] = (tree, calc_fitness(tree))
        
    best_tree, best_fit = min(population, key=lambda x: x[1])
    
    # Evolution
    while time.time() < start_time + time_budget:
        new_population = []
        
        # Elitism: Keep best
        new_population.append((best_tree, best_fit))
        
        # Generate new individuals
        while len(new_population) < pop_size:
            # Selection
            t1 = random.choice(population)
            t2 = random.choice(population)
            
            # Crossover or Mutation
            if random.random() < 0.8:
                # Crossover
                try:
                    c1, c2 = _crossover(t1[0], t2[0])
                    # Evaluate c1
                    fit1 = calc_fitness(c1)
                    new_population.append((c1, fit1))
                except Exception:
                    continue
            else:
                # Mutation
                try:
                    m1 = _mutate(t1[0], mutation_rate=0.1)
                    fit1 = calc_fitness(m1)
                    new_population.append((m1, fit1))
                except Exception:
                    continue
                    
        # Update population
        population = new_population
        # Update best
        current_best = min(population, key=lambda x: x[1])
        if current_best[1] < best_fit:
            best_fit = current_best[1]
            best_tree = current_best[0]
            
        # Early stopping if MSE is very small
        if best_fit < 1e-6:
            break

    # Create predictor using best_tree
    # The tree was evolved on x_norm.
    # Predictor:
    # 1. Normalize input x to [0, 1] using x_min, x_max from training.
    # 2. Evaluate tree.
    # 3. Denormalize y: y_pred = y_norm * y_std + y_mean.
    
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_raw = X[:, 0]
        
        if x_max == x_min:
            x_norm = np.zeros_like(x_raw)
        else:
            x_norm = (x_raw - x_min) / (x_max - x_min)
            
        try:
            y_norm_pred = best_tree.evaluate(x_norm)
            y_pred = y_norm_pred * y_std + y_mean
            # Ensure finite
            y_pred = np.where(np.isfinite(y_pred), y_pred, y_mean)
            return y_pred
        except Exception:
            return np.full(X.shape[0], y_mean)

    return predict

def fit(X_train, y_train):
    global _DEADLINE
    
    # Ensure 1D input handling
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    # Check if constant
    if np.std(y_train) < 1e-9:
        y_mean = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict

    # Time budget
    time_left = _DEADLINE - time.time()
    if time_left < 5:
        # Fallback: Linear interpolation or mean
        # Use simple linear regression as fallback
        try:
            coeffs = np.polyfit(x_train, y_train, 2)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                x_test = X[:, 0]
                return np.polyval(coeffs, x_test)
            return predict
        except Exception:
            y_mean = np.mean(y_train)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return np.full(X.shape[0], y_mean)
            return predict

    # Run GP
    predict = _fit_gp(x_train, y_train, time_budget=max(0, time_left - 5))
    return predict

# EVOLVE-BLOCK-END
