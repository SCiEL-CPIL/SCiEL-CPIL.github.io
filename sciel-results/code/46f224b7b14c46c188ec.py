"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import operator
import math
import random
from functools import reduce

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 325.0  # Safety margin for 330s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n_samples, d = X.shape
    
    # Normalize inputs to [0, 1] to improve numerical stability of trig/exp functions
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std

    # Define primitives
    # Terminals: x0, x1, ..., x_{d-1}, 1.0 (bias)
    # Functions: add, sub, mul, div, sin, cos, exp, sqrt, abs
    # We use a stack-based or tree-based representation. Tree is standard for GP.
    
    class Node:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children if children else []
            
        def depth(self):
            if not self.children:
                return 0
            return 1 + max(c.depth() for c in self.children)
            
        def size(self):
            if not self.children:
                return 1
            return 1 + sum(c.size() for c in self.children)

    # Function definitions
    def safe_div(a, b):
        # Avoid division by zero
        if np.abs(b) < 1e-8:
            return np.sign(a) * 100.0
        return a / b

    def safe_exp(a):
        # Clip to avoid overflow
        return np.exp(np.clip(a, -10, 10))

    def safe_sqrt(a):
        return np.sqrt(np.abs(a))

    def safe_log(a):
        return np.log(np.abs(a) + 1e-8)

    # Mapping of function names to (func, arity)
    FUNC_DEFS = {
        'add': (lambda a, b: a + b, 2),
        'sub': (lambda a, b: a - b, 2),
        'mul': (lambda a, b: a * b, 2),
        'div': (safe_div, 2),
        'sin': (np.sin, 1),
        'cos': (np.cos, 1),
        'exp': (safe_exp, 1),
        'sqrt': (safe_sqrt, 1),
        'abs': (np.abs, 1),
        'log': (safe_log, 1),
    }
    
    FUNC_NAMES = list(FUNC_DEFS.keys())
    
    # Terminals: x0...x_{d-1}, 1.0
    TERMINALS = [f'x{i}' for i in range(d)] + ['1.0']

    def evaluate_tree(tree, X_data):
        """
        Evaluates the tree on X_data (N, D).
        Returns array of shape (N,)
        """
        if tree.name in TERMINALS:
            if tree.name == '1.0':
                return np.ones(X_data.shape[0])
            else:
                idx = int(tree.name[1:])
                return X_data[:, idx]
        
        func, arity = FUNC_DEFS[tree.name]
        if arity == 1:
            arg = evaluate_tree(tree.children[0], X_data)
            return func(arg)
        elif arity == 2:
            arg1 = evaluate_tree(tree.children[0], X_data)
            arg2 = evaluate_tree(tree.children[1], X_data)
            return func(arg1, arg2)
        else:
            # Should not happen
            return np.zeros(X_data.shape[0])

    def generate_tree(max_depth, min_depth=0):
        """
        Generates a random tree using half-and-half method.
        """
        # Randomly decide if we start with a function or a terminal
        # To ensure we have some depth, we can bias towards functions if depth < max_depth
        # Simple random choice
        if max_depth == 0 or (min_depth > 0 and random.random() < 0.5):
             # Force function if we need depth? 
             # Actually, standard half-and-half:
             # 50% chance function, 50% chance terminal.
             # But we need to respect min_depth.
             pass
        
        # Let's use a simpler recursive generator that respects max_depth
        # and has a small chance of picking a terminal even if depth > min_depth
        # to keep trees compact.
        
        if max_depth == 0:
            return Node(random.choice(TERMINALS))
        
        # Probability of picking a terminal increases as we get deeper
        # or we just use 50/50.
        if random.random() < 0.3: # 30% chance to stop early if depth allows
             # Only if we are at min_depth or higher? 
             # Let's just say if we are at max_depth, we MUST stop.
             # If not, we can stop with some probability.
             if max_depth <= 1 or random.random() < 0.5:
                 return Node(random.choice(TERMINALS))

        func_name = random.choice(FUNC_NAMES)
        _, arity = FUNC_DEFS[func_name]
        children = []
        for _ in range(arity):
            children.append(generate_tree(max_depth - 1))
        return Node(func_name, children)

    def crossover(tree1, tree2):
        """
        Standard subtree crossover.
        Returns a new tree.
        """
        # Pick a random node from tree1 to be the root of the new tree
        # and a random node from tree2 to replace it.
        
        def get_random_node(tree):
            # Collect all nodes
            nodes = []
            def collect(t):
                nodes.append(t)
                for c in t.children:
                    collect(c)
            collect(tree)
            return random.choice(nodes)

        node1 = get_random_node(tree1)
        node2 = get_random_node(tree2)
        
        # We need to replace node1 in tree1 with node2.
        # This requires a parent pointer or a recursive copy.
        # Let's implement a recursive copy that replaces the first occurrence of node1.
        # Since node objects are unique, we can check identity.
        
        def replace_node(old_tree, target_node, new_node):
            if old_tree is target_node:
                return new_node
            new_children = []
            for c in old_tree.children:
                new_children.append(replace_node(c, target_node, new_node))
            return Node(old_tree.name, new_children)
        
        return replace_node(tree1, node1, node2)

    def mutate(tree):
        """
        Standard subtree mutation.
        """
        def get_random_node(tree):
            nodes = []
            def collect(t):
                nodes.append(t)
                for c in t.children:
                    collect(c)
            collect(tree)
            return random.choice(nodes)

        node_to_mutate = get_random_node(tree)
        new_subtree = generate_tree(max_depth=3) # Small subtree
        
        def replace_node(old_tree, target_node, new_node):
            if old_tree is target_node:
                return new_node
            new_children = []
            for c in old_tree.children:
                new_children.append(replace_node(c, target_node, new_node))
            return Node(old_tree.name, new_children)
        
        return replace_node(tree, node_to_mutate, new_subtree)

    # Initialize population
    POP_SIZE = 50
    MAX_DEPTH = 5
    population = [generate_tree(MAX_DEPTH) for _ in range(POP_SIZE)]
    
    best_tree = None
    best_fitness = float('inf')
    
    # Parsimony coefficient
    PARSIMONY = 0.01
    
    def get_fitness(tree, X_data, y_data):
        """
        Calculates fitness. Lower is better.
        Uses RMSE + Parsimony * Size
        """
        try:
            pred = evaluate_tree(tree, X_data)
            # Check for NaNs or Infs
            if not np.all(np.isfinite(pred)):
                return float('inf')
            
            mse = np.mean((pred - y_data) ** 2)
            size = tree.size()
            return mse + PARSIMONY * size
        except Exception:
            return float('inf')

    # Initial evaluation
    for tree in population:
        f = get_fitness(tree, X_norm, y_norm)
        if f < best_fitness:
            best_fitness = f
            best_tree = tree

    # Evolution Loop
    GENERATIONS = 100 # Max generations, but limited by time
    
    gen = 0
    while gen < GENERATIONS:
        if time.time() > DEADLINE - 5.0:
            break
            
        # Selection: Tournament selection
        def tournament_select(k=5):
            candidates = random.sample(population, k)
            best_cand = None
            best_f = float('inf')
            for c in candidates:
                # We should cache fitness? 
                # For simplicity and memory, re-evaluate or store in a dict?
                # Storing in a list parallel to population is better.
                pass
            
            # Let's maintain a fitness list
            # This makes the loop structure slightly more complex but faster.
            # For now, re-evaluating is O(N * Size). N=50, Size~10. 500 ops per tournament.
            # 50 tournaments per gen. 25000 ops. Very fast.
            
            best_cand = min(candidates, key=lambda t: get_fitness(t, X_norm, y_norm))
            return best_cand

        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_tree)
        
        while len(new_population) < POP_SIZE:
            if time.time() > DEADLINE - 5.0:
                break
                
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            r = random.random()
            if r < 0.8: # Crossover
                child = crossover(parent1, parent2)
            else: # Mutation
                child = mutate(parent1)
            
            new_population.append(child)
        
        population = new_population
        
        # Update best
        for tree in population:
            f = get_fitness(tree, X_norm, y_norm)
            if f < best_fitness:
                best_fitness = f
                best_tree = tree
        
        gen += 1

    # Final Evaluation and Prediction Function
    if best_tree is None:
        # Fallback: Linear regression
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            # Simple linear fit
            A = np.hstack([X_test_norm, np.ones((X_test_norm.shape[0], 1))])
            # We need weights. Let's just return mean for safety if GP failed completely
            return np.full(X_test.shape[0], y_mean)
        return predict_fallback

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        try:
            pred_norm = evaluate_tree(best_tree, X_test_norm)
            # Ensure finite
            pred_norm = np.where(np.isfinite(pred_norm), pred_norm, 0.0)
            return pred_norm * y_std + y_mean
        except Exception:
            return np.full(X_test.shape[0], y_mean)

    return predict
# EVOLVE-BLOCK-END
