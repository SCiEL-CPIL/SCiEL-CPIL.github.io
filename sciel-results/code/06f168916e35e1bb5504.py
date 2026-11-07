"""Genetic Programming Symbolic Regression with Progressive Complexity for 1-D Noiseless Data."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
from operator import add, sub, mul, truediv

# Deadline guard: 330s wall-clock limit, use 280s for safety to leave room for final prediction
_DEADLINE = time.time() + 280.0

def _remaining_time():
    return max(0.0, _DEADLINE - time.time())

class GPNode:
    __slots__ = ['op', 'left', 'right']
    
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right
    
    def __repr__(self):
        if self.left is None:
            return f"{self.op}({self.left})"
        return f"{self.op}({self.left}, {self.right})"

def _safe_div(x, y):
    return np.where(np.abs(y) < 1e-10, 1e-10, x / y)

def _safe_log(x):
    return np.log(np.abs(x) + 1e-10)

def _safe_sqrt(x):
    return np.sqrt(np.abs(x))

def _safe_exp(x):
    # Clip to avoid overflow
    return np.exp(np.clip(x, -50, 50))

# Terminal and Function definitions
TERMINALS = ['x', 'pi']
FUNCTIONS = {
    'add': (add, 2),
    'sub': (sub, 2),
    'mul': (mul, 2),
    'div': (_safe_div, 2),
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'tan': (np.tan, 1),
    'exp': (_safe_exp, 1),
    'log': (_safe_log, 1),
    'sqrt': (_safe_sqrt, 1),
    'abs': (np.abs, 1),
}

def _create_random_tree(depth, max_depth, rng):
    """Create a random expression tree."""
    if depth >= max_depth or (rng.random() < 0.1 and depth > 0):
        # Terminal
        term = rng.choice(TERMINALS)
        if term == 'pi':
            return GPNode('const', np.pi)
        else:
            return GPNode('x')
    
    # Function node
    func_name, arity = rng.choice(list(FUNCTIONS.items()))
    op_func, _ = FUNCTIONS[func_name]
    
    if arity == 1:
        left = _create_random_tree(depth + 1, max_depth, rng)
        return GPNode(func_name, left, None)
    else:
        left = _create_random_tree(depth + 1, max_depth, rng)
        right = _create_random_tree(depth + 1, max_depth, rng)
        return GPNode(func_name, left, right)

def _evaluate_tree(node, x):
    """Evaluate a GP tree on input x."""
    op = node.op
    
    if op == 'x':
        return x
    elif op == 'const':
        return np.full_like(x, node.left)
    elif op == 'add':
        return _evaluate_tree(node.left, x) + _evaluate_tree(node.right, x)
    elif op == 'sub':
        return _evaluate_tree(node.left, x) - _evaluate_tree(node.right, x)
    elif op == 'mul':
        return _evaluate_tree(node.left, x) * _evaluate_tree(node.right, x)
    elif op == 'div':
        return _safe_div(_evaluate_tree(node.left, x), _evaluate_tree(node.right, x))
    elif op in FUNCTIONS:
        func, _ = FUNCTIONS[op]
        if op in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']:
            return func(_evaluate_tree(node.left, x))
        else:
            return _evaluate_tree(node.left, x)
    else:
        return np.zeros_like(x)

def _get_subtree(node, rng):
    """Get a random subtree index."""
    if node.left is None:
        return 0
    # 50% chance left, 50% chance right (simplified)
    return rng.choice([0, 1])

def _crossover(parent1, parent2, rng):
    """Crossover between two trees."""
    # For simplicity, we'll just swap a random subtree.
    # A more robust implementation would traverse to a random node.
    # Here we use a simplified approach: replace a random part of parent1 with parent2's root
    # or vice versa. This is a crude crossover but works for small trees.
    
    # Better: Pick a random node in parent1 and replace it with parent2
    # We need a helper to pick a random node and replace it.
    
    def _replace_random_subtree(node, replacement, rng):
        if node.left is None:
            # Leaf, can't replace with subtree easily in this structure unless we change type
            # Just return node for now, or if replacement is also leaf, swap.
            return node
        # Decide whether to replace this node's left or right, or recurse
        # If we recurse, we might go too deep.
        # Let's just swap the entire right child of a random internal node.
        # This is a placeholder for a proper crossover.
        pass
        
    # Simple Crossover: Return a copy of parent1 with a random node replaced by parent2
    # Since implementing a full tree traversal for replacement is complex in this slot-based structure,
    # we will use a simpler mutation-like crossover:
    # With 50% chance, return parent1. With 50% chance, return parent2.
    # Or, better, create a new tree that combines parts.
    
    # Let's do a proper recursive crossover.
    def _crossover_recursive(node, other, rng):
        # Choose a random node in 'node' to replace with 'other'
        # We'll pick a depth and then a path.
        # Simplified: just return other if rng.random() < 0.5 else node? No, that's not crossover.
        
        # Let's implement a proper subtree swap.
        # We need to find a random node in the tree.
        # We'll do a DFS and pick a random node from the list of nodes.
        
        nodes = []
        def _collect(node):
            if node is None:
                return
            nodes.append(node)
            if node.left is not None:
                _collect(node.left)
                if node.right is not None:
                    _collect(node.right)
        
        _collect(node)
        
        if not nodes:
            return node
            
        # Pick a random node to replace
        # But we can only replace internal nodes with subtrees, or leaves with leaves.
        # To keep it simple and safe, we will only replace internal nodes (nodes with children)
        # with the 'other' tree.
        
        internal_nodes = [n for n in nodes if n.left is not None]
        
        if not internal_nodes:
            return node
            
        target = rng.choice(internal_nodes)
        
        # We need to replace 'target' in the original tree with 'other'.
        # This requires a recursive replacement function.
        
        def _replace(node, target, replacement):
            if node is target:
                return replacement
            if node.left is not None:
                node.left = _replace(node.left, target, replacement)
                if node.right is not None:
                    node.right = _replace(node.right, target, replacement)
            return node
            
        # Deep copy is not strictly necessary if we don't mutate the originals in a way that affects other parts,
        # but since we are modifying pointers, we should be careful.
        # For now, we'll assume we can modify the tree in-place for the new individual.
        # However, GP usually works on copies.
        
        # Let's just return the modified tree.
        return _replace(node, target, other)

    return _crossover_recursive(parent1, parent2, rng)

def _mutate(node, rng, max_depth):
    """Mutate a tree by replacing a random subtree with a new random tree."""
    nodes = []
    def _collect(node):
        if node is None:
            return
        nodes.append(node)
        if node.left is not None:
            _collect(node.left)
            if node.right is not None:
                _collect(node.right)
    
    _collect(node)
    
    if not nodes:
        return node
        
    target = rng.choice(nodes)
    
    # Create a new random subtree to replace target
    # The depth of the new subtree should be limited to avoid explosion
    new_depth = rng.randint(0, 2)
    new_subtree = _create_random_tree(0, new_depth, rng)
    
    def _replace(node, target, replacement):
        if node is target:
            return replacement
        if node.left is not None:
            node.left = _replace(node.left, target, replacement)
            if node.right is not None:
                node.right = _replace(node.right, target, replacement)
        return node
        
    return _replace(node, target, new_subtree)

def _fit_gp(X_train, y_train):
    """
    Main GP fit function.
    """
    x_train = X_train[:, 0]
    y_train = y_train
    
    rng = random.Random(42) # Fixed seed for reproducibility within the run? Or random seed?
    # Let's use a random seed based on time to allow different runs to find different solutions if needed,
    # but for a single fit call, it's deterministic.
    
    population_size = 50
    max_generations = 200
    max_depth = 4
    
    # Initialize population
    population = []
    for _ in range(population_size):
        depth = rng.randint(1, 3)
        tree = _create_random_tree(0, depth, rng)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    
    # Evaluate initial population
    for tree in population:
        try:
            pred = _evaluate_tree(tree, x_train)
            mse = np.mean((pred - y_train) ** 2)
            if mse < best_fitness:
                best_fitness = mse
                best_tree = tree
        except Exception:
            continue
            
    # Evolution loop
    gen = 0
    while gen < max_generations and _remaining_time() > 5.0:
        gen += 1
        
        new_population = []
        
        # Elitism: Keep the best individual
        if best_tree is not None:
            new_population.append(best_tree)
            
        # Generate rest of population
        while len(new_population) < population_size:
            if rng.random() < 0.7:
                # Crossover
                p1 = rng.choice(population)
                p2 = rng.choice(population)
                child = _crossover(p1, p2, rng)
            else:
                # Mutation
                p1 = rng.choice(population)
                child = _mutate(p1, rng, max_depth)
                
            new_population.append(child)
            
        # Evaluate new population
        for tree in new_population:
            try:
                pred = _evaluate_tree(tree, x_train)
                mse = np.mean((pred - y_train) ** 2)
                if mse < best_fitness:
                    best_fitness = mse
                    best_tree = tree
            except Exception:
                continue
                
        population = new_population
        
        # If we've found a very good fit, we might stop early
        if best_fitness < 1e-10:
            break

    return best_tree, best_fitness

def fit(X_train, y_train):
    """
    Fit the model using Genetic Programming.
    """
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    
    x_train = X_train[:, 0]
    y_train = y_train
    
    # Try GP
    best_tree, best_mse = _fit_gp(X_train, y_train)
    
    # Fallback: If GP fails or takes too long, use a simple interpolator
    if best_tree is None or best_mse > 1.0: # Threshold for "bad" fit
        # Use PCHIP as a robust fallback
        from scipy.interpolate import PchipInterpolator
        sort_idx = np.argsort(x_train)
        x_sorted = x_train[sort_idx]
        y_sorted = y_train[sort_idx]
        try:
            pchip = PchipInterpolator(x_sorted, y_sorted)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                x_test = X[:, 0]
                preds = pchip(x_test)
                preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
                return preds
            return predict
        except Exception:
            y_mean = np.mean(y_train)
            def predict(X):
                if X.ndim == 1:
                    X = X.reshape(-1, 1)
                return np.full(X.shape[0], y_mean)
            return predict
            
    # Wrap the best tree in a predict function
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_test = X[:, 0]
        try:
            preds = _evaluate_tree(best_tree, x_test)
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            return preds
        except Exception:
            return np.full(X.shape[0], np.mean(y_train))
            
    return predict

# EVOLVE-BLOCK-END
