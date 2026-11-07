"""
Genetic Programming (GP) Symbolic Regression for 1-D Noiseless Equations.

Strategy:
1. The problem is 1-D symbolic regression on 4 hidden equations.
2. Instead of assuming a specific functional form (like Chebyshev polynomials),
   we use Genetic Programming to evolve mathematical expressions from a set of
   primitives.
3. The fitness function is Negative Mean Squared Error (NMSE). Since data is noiseless,
   a perfect fit yields MSE ~ 0.
4. We run a time-boxed GP evolution. If a perfect or near-perfect fit is found,
   we use the evolved expression for prediction.
5. If the deadline is reached without a perfect fit, we fall back to a high-capacity
   RBF (Radial Basis Function) interpolator, which can fit any 1-D function well
   given enough basis points, serving as a robust "catch-all" for cases where GP
   fails to find the exact symbolic form.

This approach is structurally different from the current Chebyshev approach:
- Chebyshev: Fixed basis, assumes smoothness, fails on exp/sin/log/abs if degree isn't high enough or if function is non-analytic.
- GP: Flexible basis, searches for the exact structure, can handle non-smooth and transcendental functions.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import operator
from collections import defaultdict

# Set a global deadline to ensure we finish within the 330s limit (with margin)
# The evaluator kills at 330s. We aim to finish logic by 320s.
_DEADLINE = time.time() + 320.0

# --- GP Primitives ---
# We define a set of terminal and non-terminal functions.
# Terminals: x (input)
# Non-terminals: +, -, *, /, sin, cos, exp, log, abs, sqrt, x^2, x^3

def _safe_div(a, b):
    """Safe division to avoid division by zero or overflow."""
    # If b is close to 0, return a large value or 0? 
    # In symbolic regression, we usually want to penalize instability.
    # Let's return 0 where b is 0, but this can be tricky.
    # Better: use np.where to handle it, but we are in GP evaluation.
    # For GP, we can clip the denominator.
    eps = 1e-8
    b_safe = np.where(np.abs(b) < eps, eps * np.sign(b), b)
    # Handle sign(0)
    b_safe = np.where(b_safe == 0, eps, b_safe)
    return a / b_safe

def _safe_exp(a):
    """Safe exponential to avoid overflow."""
    # Clip input to prevent inf
    return np.exp(np.clip(a, -50, 50))

def _safe_log(a):
    """Safe log to avoid log(0) or log(negative)."""
    # For symbolic regression, usually we assume domain where log is valid,
    # but to be robust, we take log(abs(a)) or clip.
    # Given the input ranges, x can be negative. log(x) would be invalid.
    # The hidden equations likely use valid operations.
    # If the equation is log(x), x must be positive.
    # If the equation is log(abs(x)), we need abs.
    # Let's use log(abs(a) + eps) to be safe against negatives/zero, 
    # but this changes the function. 
    # Alternative: If the true function is log(x), and x>0, then log(x) is fine.
    # If we use log(abs(x)), we might fit log(x) for x>0 but fail for x<0 if true func is undefined there.
    # However, the input ranges include negatives for some equations.
    # Equation 4 is [0.1, 6], so log(x) is safe there.
    # Equations 1-3 are [-3,3] or [-2,2]. If they involve log, they likely use log(abs(x)) or similar.
    # Let's stick to standard log but handle errors by returning a large penalty in fitness.
    # Actually, for the *predictor*, we just need to return finite values.
    # Let's use np.log(np.abs(a) + 1e-8) as a robust approximation.
    return np.log(np.abs(a) + 1e-8)

def _safe_sqrt(a):
    """Safe sqrt."""
    return np.sqrt(np.abs(a) + 1e-8)

# Define the function map
FUNCTIONS = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': _safe_div,
    'sin': np.sin,
    'cos': np.cos,
    'exp': _safe_exp,
    'log': _safe_log,
    'abs': np.abs,
    'sqrt': _safe_sqrt,
    'pow2': lambda x: x * x,
    'pow3': lambda x: x * x * x,
}

# Terminal: just 'x'

class GPNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children else []

    def evaluate(self, x):
        """Recursively evaluate the tree."""
        if self.name == 'x':
            return x
        if self.name in FUNCTIONS:
            if len(self.children) == 1:
                return FUNCTIONS[self.name](self.children[0].evaluate(x))
            elif len(self.children) == 2:
                return FUNCTIONS[self.name](self.children[0].evaluate(x), self.children[1].evaluate(x))
            else:
                raise ValueError(f"Function {self.name} has {len(self.children)} children")
        else:
            raise ValueError(f"Unknown node {self.name}")

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def clone(self):
        """Deep copy of the tree."""
        if not self.children:
            return GPNode(self.name)
        return GPNode(self.name, [c.clone() for c in self.children])

    def __str__(self):
        if not self.children:
            return self.name
        return f"({self.name} " + " ".join(str(c) for c in self.children) + ")"

def _create_random_tree(max_depth, func_prob=0.7):
    """Create a random GP tree."""
    if max_depth == 1 or random.random() > func_prob:
        # Terminal
        return GPNode('x')
    
    # Choose a function
    # Unary functions: sin, cos, exp, log, abs, sqrt, pow2, pow3
    # Binary functions: add, sub, mul, div
    # Let's pick randomly from all, but respect arity.
    # To simplify, let's define specific probabilities.
    # 50% chance binary, 50% chance unary?
    # Or just pick from a weighted list.
    
    func_names = list(FUNCTIONS.keys())
    # We need to know arity.
    # Let's just pick a random function name and handle arity in creation.
    # Actually, it's safer to define unary and binary lists separately.
    
    unary_funcs = ['sin', 'cos', 'exp', 'log', 'abs', 'sqrt', 'pow2', 'pow3']
    binary_funcs = ['add', 'sub', 'mul', 'div']
    
    if random.random() < 0.5:
        # Unary
        name = random.choice(unary_funcs)
        child = _create_random_tree(max_depth - 1, func_prob)
        return GPNode(name, [child])
    else:
        # Binary
        name = random.choice(binary_funcs)
        child1 = _create_random_tree(max_depth - 1, func_prob)
        child2 = _create_random_tree(max_depth - 1, func_prob)
        return GPNode(name, [child1, child2])

def _mutate_tree(tree, mutation_rate=0.1):
    """Mutate a tree by replacing a random subtree."""
    # Flatten tree to list of nodes for random selection
    nodes = []
    def _collect(node):
        nodes.append(node)
        for c in node.children:
            _collect(c)
    _collect(tree)
    
    if not nodes:
        return tree
    
    # Pick a random node to mutate
    target_node = random.choice(nodes)
    
    # Determine max depth allowed for the new subtree
    # This is a rough heuristic. We can just generate a small tree.
    max_depth = 3
    
    # If the target node is a function, we can replace it with a new subtree of the same arity.
    # If it's a terminal, we can replace it with a small subtree.
    
    # For simplicity, we will replace the target node with a newly generated random tree.
    # We need to preserve arity if the parent expects a specific number of children?
    # No, in GP, a node replaces a whole subtree. The parent just holds a reference to the root of the subtree.
    # So we can replace any node with any tree.
    
    new_subtree = _create_random_tree(max_depth, func_prob=0.8)
    
    # Replace target_node with new_subtree in the parent
    # We need to find the parent.
    # Easier: just rebuild the tree? No, that's slow.
    # Let's do a recursive replacement.
    
    def _replace(node, target):
        if node is target:
            return new_subtree
        new_children = []
        for c in node.children:
            new_children.append(_replace(c, target))
        node.children = new_children
        return node
    
    _replace(tree, target_node)
    return tree

def _crossover(tree1, tree2):
    """Crossover between two trees."""
    # Pick a random node from tree1 and a random node from tree2
    # Swap them.
    
    nodes1 = []
    def _collect1(node):
        nodes1.append(node)
        for c in node.children:
            _collect1(c)
    _collect1(tree1)
    
    nodes2 = []
    def _collect2(node):
        nodes2.append(node)
        for c in node.children:
            _collect2(c)
    _collect2(tree2)
    
    if not nodes1 or not nodes2:
        return tree1.clone(), tree2.clone()
    
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)
    
    # Clone trees to avoid side effects
    t1 = tree1.clone()
    t2 = tree2.clone()
    
    # We need to find the original nodes in the cloned trees to swap.
    # This is tricky because clone creates new objects.
    # Instead, let's do the swap on the original trees and then clone the results?
    # No, let's just implement a swap that works on references.
    
    # Find node1 in t1? No, t1 is a clone.
    # Let's re-implement:
    # 1. Find a random node in tree1 (original)
    # 2. Find a random node in tree2 (original)
    # 3. Create child1 = tree1.clone() and replace the corresponding node with tree2's node clone?
    # This is getting complicated.
    
    # Simpler Crossover:
    # 1. Pick a random subtree from tree1.
    # 2. Pick a random subtree from tree2.
    # 3. Swap them in the original trees? No, we want two new trees.
    
    # Let's just do:
    # child1 = tree1.clone()
    # child2 = tree2.clone()
    # Find a random node in child1 and replace it with a clone of a random node from tree2.
    # Find a random node in child2 and replace it with a clone of a random node from tree1.
    
    # To find a random node in a cloned tree, we can just pick an index from the flattened list of the ORIGINAL tree,
    # assuming the structure is identical (which it is, since it's a clone).
    
    # Flatten original trees
    flat1 = []
    def _f1(n):
        flat1.append(n)
        for c in n.children: _f1(c)
    _f1(tree1)
    
    flat2 = []
    def _f2(n):
        flat2.append(n)
        for c in n.children: _f2(c)
    _f2(tree2)
    
    idx1 = random.randint(0, len(flat1)-1)
    idx2 = random.randint(0, len(flat2)-1)
    
    # The node to swap in tree1 is flat1[idx1]. In the clone, it's at the same structural position.
    # We can just clone the node from tree2 and replace it in tree1's clone.
    
    c1 = tree1.clone()
    c2 = tree2.clone()
    
    # We need to find the corresponding node in c1 and c2.
    # Since clone preserves structure, we can traverse both simultaneously.
    
    def _swap(node_c1, node_c2, target1_idx, target2_idx, curr1=0, curr2=0):
        # This is messy. Let's use a different approach.
        # Just pick a random node in c1 and replace it with a random node from c2 (cloned).
        pass

    # Let's simplify Crossover:
    # 1. Pick a random node from tree1.
    # 2. Pick a random node from tree2.
    # 3. child1 = tree1.clone()
    # 4. Replace the chosen node in child1 with a clone of the chosen node from tree2.
    # 5. child2 = tree2.clone()
    # 6. Replace the chosen node in child2 with a clone of the chosen node from tree1.
    
    # To replace a specific node in a clone, we need to know its path or use identity.
    # Since we can't easily map original node to clone node by identity, let's do this:
    
    # Pick random subtree from tree1 (by cloning a random node from a random position)
    # Actually, let's just pick a random node from the *original* trees, clone it, and insert it.
    
    # But we need to remove the old node.
    
    # Let's implement a helper that returns a new tree where a specific random node is replaced.
    
    def _replace_random_node(tree, replacement_node):
        """Returns a new tree where a random node is replaced by replacement_node."""
        # Flatten
        nodes = []
        def _col(n):
            nodes.append(n)
            for c in n.children: _col(c)
        _col(tree)
        
        if not nodes:
            return tree.clone()
        
        target = random.choice(nodes)
        
        # Build new tree
        def _build(n):
            if n is target:
                return replacement_node.clone()
            return GPNode(n.name, [_build(c) for c in n.children])
        
        return _build(tree)
    
    # Pick random nodes from original trees to use as replacements
    rep1 = random.choice(flat1).clone()
    rep2 = random.choice(flat2).clone()
    
    child1 = _replace_random_node(tree1, rep2)
    child2 = _replace_random_node(tree2, rep1)
    
    return child1, child2

def _calculate_fitness(tree, x, y):
    """Calculate NMSE. Lower is better. We return negative MSE for maximization or just MSE for minimization."""
    try:
        pred = tree.evaluate(x)
        # Check for non-finite values
        if not np.all(np.isfinite(pred)):
            return np.inf
        mse = np.mean((pred - y) ** 2)
        return mse
    except Exception:
        return np.inf

def _evolve_gp(x, y, population_size=100, generations=200, elite_size=5, tournament_size=5):
    """Run GP evolution."""
    # Initialize population
    population = []
    for _ in range(population_size):
        tree = _create_random_tree(3, func_prob=0.7)
        population.append(tree)
    
    best_tree = None
    best_fitness = np.inf
    
    gen = 0
    while gen < generations:
        if time.time() > _DEADLINE - 10:
            break
            
        # Evaluate fitness
        fitnesses = [_calculate_fitness(t, x, y) for t in population]
        
        # Update best
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_fitness:
            best_fitness = fitnesses[min_idx]
            best_tree = population[min_idx].clone()
            
        # If perfect fit found, stop
        if best_fitness < 1e-10:
            break
            
        # Selection (Tournament)
        def _select(pop, fit):
            indices = [random.randint(0, len(pop)-1) for _ in range(tournament_size)]
            best_idx = min(indices, key=lambda i: fit[i])
            return pop[best_idx]
        
        # Create next generation
        next_population = []
        
        # Elitism
        sorted_indices = np.argsort(fitnesses)
        for i in range(elite_size):
            next_population.append(population[sorted_indices[i]].clone())
            
        # Fill rest with offspring
        while len(next_population) < population_size:
            if time.time() > _DEADLINE - 5:
                break
                
            parent1 = _select(population, fitnesses)
            parent2 = _select(population, fitnesses)
            
            # Crossover or Mutation
            if random.random() < 0.7:
                c1, c2 = _crossover(parent1, parent2)
                # Mutate children
                if random.random() < 0.1:
                    c1 = _mutate_tree(c1)
                if random.random() < 0.1:
                    c2 = _mutate_tree(c2)
                next_population.append(c1)
                next_population.append(c2)
            else:
                c1 = parent1.clone()
                c2 = parent2.clone()
                c1 = _mutate_tree(c1)
                c2 = _mutate_tree(c2)
                next_population.append(c1)
                next_population.append(c2)
                
        population = next_population
        gen += 1
        
    return best_tree, best_fitness

def _build_rbf_fallback(x_train, y_train):
    """
    Build a high-capacity RBF interpolator as a fallback.
    Uses all training points as centers.
    Solves for weights using ridge regression.
    """
    x = x_train
    y = y_train
    n = len(x)
    
    # Use a small bandwidth for interpolation
    # Sigma = distance between adjacent points / 2? Or fixed?
    # Let's use a small fixed sigma relative to domain size.
    domain_size = np.max(x) - np.min(x)
    if domain_size < 1e-6:
        domain_size = 1.0
    sigma = domain_size / (n * 2)
    
    # Construct RBF matrix
    # K[i, j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    X_diff = x[:, np.newaxis] - x[np.newaxis, :]
    K = np.exp(-X_diff ** 2 / (2 * sigma ** 2))
    
    # Regularize
    lam = 1e-6
    K_reg = K + lam * np.eye(n)
    
    # Solve for weights
    try:
        weights = np.linalg.solve(K_reg, y)
    except np.linalg.LinAlgError:
        # Fallback to pinv
        weights = np.linalg.pinv(K_reg) @ y
        
    def predict(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        x_test = X[:, 0]
        X_diff = x_test[:, np.newaxis] - x[np.newaxis, :]
        K_test = np.exp(-X_diff ** 2 / (2 * sigma ** 2))
        preds = K_test @ weights
        return np.where(np.isfinite(preds), preds, np.mean(y))
        
    return predict

def fit(X_train, y_train):
    global _DEADLINE
    
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        
    x_train = X_train[:, 0]
    y_train = y_train
    
    n = len(x_train)
    if n == 0:
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.zeros(X.shape[0])
        return predict

    if np.max(x_train) - np.min(x_train) < 1e-10:
        y_mean = np.mean(y_train)
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            return np.full(X.shape[0], y_mean)
        return predict

    # Try GP
    best_tree = None
    best_fitness = np.inf
    
    try:
        # Use a reasonable budget for GP. 
        # If we have 300s total, and we have 4 equations, we have ~75s per equation.
        # But the deadline is global. 
        # Let's allocate a fixed time slice or check global deadline.
        # We will run GP until deadline - 5s.
        
        # To prevent one equation from eating all time, we can limit generations
        # or just rely on the global deadline check inside _evolve_gp.
        
        best_tree, best_fitness = _evolve_gp(x_train, y_train, population_size=50, generations=500)
        
    except Exception:
        pass
        
    if best_tree is not None and best_fitness < 1e-8:
        # Perfect or near-perfect fit found
        def predict(X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            x_test = X[:, 0]
            preds = best_tree.evaluate(x_test)
            # Ensure finite
            preds = np.where(np.isfinite(preds), preds, np.mean(y_train))
            return preds
        return predict
    else:
        # Fallback to RBF
        return _build_rbf_fallback(x_train, y_train)

# EVOLVE-BLOCK-END
