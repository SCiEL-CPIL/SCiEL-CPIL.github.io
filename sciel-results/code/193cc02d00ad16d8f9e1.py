"""Symbolic regression via genetic programming with time-aware budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import product


def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = start_time + 280.0  # 50s margin for 330s limit
    
    n, d = X_train.shape
    y = y_train
    y_mean = np.mean(y)
    y_std = np.std(y)
    
    if y_std < 1e-10:
        return lambda X: np.full(len(X), y_mean)
    
    # Normalize inputs to [0, 1] range based on training data
    x_min = np.min(X_train, axis=0)
    x_max = np.max(X_train, axis=0)
    x_range = x_max - x_min + 1e-8  # avoid division by zero
    
    def normalize(X):
        return (X - x_min) / x_range
    
    X_norm = normalize(X_train)
    
    # Basic operations
    def add(a, b):
        return a + b
    def sub(a, b):
        return a - b
    def mul(a, b):
        return a * b
    def div(a, b):
        return a / (b + 1e-8 * np.sign(b))
    def neg(a):
        return -a
    def sqrt(a):
        return np.sqrt(np.abs(a) + 1e-8)
    def exp(a):
        return np.exp(np.clip(a, -20, 20))
    def log(a):
        return np.log(np.abs(a) + 1e-8)
    def sin(a):
        return np.sin(a)
    def cos(a):
        return np.cos(a)
    
    # Function registry
    functions = {
        'add': add,
        'sub': sub,
        'mul': mul,
        'div': div,
        'neg': neg,
        'sqrt': sqrt,
        'exp': exp,
        'log': log,
        'sin': sin,
        'cos': cos
    }
    
    # Terminal nodes: input variables x0, x1, ... and constants
    terminals = list(range(d))  # input indices
    # Add some random constants as terminals
    constants = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    for c in constants:
        terminals.append(('const', c))
    
    def eval_tree(tree, X):
        """Recursively evaluate a tree on input X (normalized)."""
        if isinstance(tree, tuple):
            if tree[0] == 'const':
                return np.full(len(X), tree[1])
            else:
                op = tree[0]
                args = tree[1:]
                if len(args) == 1:
                    return functions[op](eval_tree(args[0], X))
                else:
                    return functions[op](eval_tree(args[0], X), eval_tree(args[1], X))
        else:
            # tree is an integer index into X
            return X[:, tree]
    
    def random_tree(depth=0, max_depth=4):
        """Generate a random expression tree."""
        if depth >= max_depth:
            return random.choice(terminals)
        
        # Choose between function and terminal with some probability
        if random.random() < 0.3 or depth == max_depth:
            return random.choice(terminals)
        
        # Choose a function
        op = random.choice(list(functions.keys()))
        
        if op in ['neg', 'sqrt', 'exp', 'log', 'sin', 'cos']:
            return (op, random_tree(depth + 1, max_depth))
        else:
            return (op, random_tree(depth + 1, max_depth), random_tree(depth + 1, max_depth))
    
    def tree_depth(tree):
        if isinstance(tree, tuple):
            if tree[0] == 'const':
                return 1
            else:
                args = tree[1:]
                return 1 + max(tree_depth(a) for a in args)
        else:
            return 1
    
    def crossover(tree1, tree2):
        """Crossover between two trees."""
        # Get all nodes in tree1
        def get_nodes(tree, path=None):
            if path is None:
                path = []
            nodes = [path]
            if isinstance(tree, tuple):
                if tree[0] == 'const':
                    return nodes
                else:
                    args = tree[1:]
                    for i, arg in enumerate(args):
                        nodes.extend(get_nodes(arg, path + [i]))
            return nodes
        
        # Randomly pick a subtree from tree1 and replace with subtree from tree2
        def get_subtree(tree, path):
            for p in path:
                tree = tree[p + 1]  # skip op name
            return tree
        
        nodes1 = get_nodes(tree1)
        nodes2 = get_nodes(tree2)
        
        if not nodes1 or not nodes2:
            return tree1
        
        path1 = random.choice(nodes1)
        path2 = random.choice(nodes2)
        
        # Replace subtree
        def replace_subtree(tree, path, new_subtree):
            if not path:
                return new_subtree
            if isinstance(tree, tuple):
                if tree[0] == 'const':
                    return tree
                args = tree[1:]
                new_args = []
                for i, arg in enumerate(args):
                    if path[0] == i:
                        new_args.append(replace_subtree(arg, path[1:], new_subtree))
                    else:
                        new_args.append(arg)
                return (tree[0],) + tuple(new_args)
            else:
                return tree
        
        subtree2 = get_subtree(tree2, path2)
        return replace_subtree(tree1, path1, subtree2)
    
    def mutate(tree, rate=0.1):
        """Mutate a tree by replacing a random subtree."""
        def get_nodes(tree, path=None):
            if path is None:
                path = []
            nodes = [path]
            if isinstance(tree, tuple):
                if tree[0] == 'const':
                    return nodes
                else:
                    args = tree[1:]
                    for i, arg in enumerate(args):
                        nodes.extend(get_nodes(arg, path + [i]))
            return nodes
        
        def replace_subtree(tree, path, new_subtree):
            if not path:
                return new_subtree
            if isinstance(tree, tuple):
                if tree[0] == 'const':
                    return tree
                args = tree[1:]
                new_args = []
                for i, arg in enumerate(args):
                    if path[0] == i:
                        new_args.append(replace_subtree(arg, path[1:], new_subtree))
                    else:
                        new_args.append(arg)
                return (tree[0],) + tuple(new_args)
            else:
                return tree
        
        nodes = get_nodes(tree)
        if not nodes:
            return tree
        
        if random.random() < rate:
            path = random.choice(nodes)
            new_subtree = random_tree(depth=0, max_depth=min(3, tree_depth(tree) - len(path)))
            return replace_subtree(tree, path, new_subtree)
        return tree
    
    def fitness(tree):
        """Compute normalized MSE for a tree."""
        try:
            pred = eval_tree(tree, X_norm)
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y)**2)
            return mse / (y_std**2)
        except Exception:
            return np.inf
    
    # Initialize population
    pop_size = 50
    population = [random_tree(depth=0, max_depth=4) for _ in range(pop_size)]
    
    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]
    best_idx = np.argmin(fitnesses)
    best_tree = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    # Genetic algorithm loop
    gen = 0
    while time.time() < DEADLINE:
        gen += 1
        
        # Select parents using tournament selection
        def tournament_select(pop, fits, k=3):
            indices = random.sample(range(len(pop)), k)
            best_i = min(indices, key=lambda i: fits[i])
            return pop[best_i]
        
        new_population = []
        
        # Elitism: keep best individual
        new_population.append(best_tree)
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > DEADLINE:
                break
            
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            
            r = random.random()
            if r < 0.5:
                child = crossover(parent1, parent2)
            elif r < 0.8:
                child = mutate(parent1)
            else:
                child = random_tree(depth=0, max_depth=4)
            
            # Limit tree size
            if tree_depth(child) > 8:
                child = random_tree(depth=0, max_depth=3)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [fitness(ind) for ind in new_population]
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fitness:
            best_tree = new_population[best_idx]
            best_fitness = fitnesses[best_idx]
        
        population = new_population
        
        # If best fitness is very good, stop early
        if best_fitness < 1e-6:
            break
    
    # Final prediction function
    def predict(X):
        X_norm_pred = normalize(X)
        try:
            pred = eval_tree(best_tree, X_norm_pred)
            if not np.all(np.isfinite(pred)):
                return np.full(len(X), y_mean)
            return pred
        except Exception:
            return np.full(len(X), y_mean)
    
    return predict
# EVOLVE-BLOCK-END
