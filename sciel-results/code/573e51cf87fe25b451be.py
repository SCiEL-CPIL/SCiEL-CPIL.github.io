"""Symbolic regression via evolutionary expression search + learned NN fallback."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import copy

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin before 330s kill
    
    d = X_train.shape[1]
    n = len(X_train)
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    
    if y_var < 1e-12:
        return lambda X: np.full(X.shape[0], y_mean)
    
    # Normalize inputs for numerical stability in GP/NN
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X_train - X_mean) / X_std
    
    y_std = np.std(y_train)
    y_norm = (y_train - y_mean) / y_std
    
    def denorm_predict(pred_norm):
        return pred_norm * y_std + y_mean
    
    def norm_mse(pred, y):
        return np.mean((pred - y) ** 2) / y_var
    
    best_score = np.inf
    best_predictor = None
    
    def _try_update(model_fn):
        nonlocal best_score, best_predictor
        try:
            pred = model_fn(X_train)
            if not np.all(np.isfinite(pred)):
                return
            sc = norm_mse(pred, y_train)
            if sc < best_score:
                best_score = sc
                best_predictor = model_fn
        except Exception:
            pass
    
    # --- Strategy 1: Genetic Programming for symbolic expressions ---
    def make_expression_tree(depth=3, d_input=d, rng=None):
        """Create a random expression tree."""
        if rng is None:
            rng = random
        if depth == 0:
            # Leaf: input variable or constant
            if rng.random() < 0.7:
                return ('var', rng.randint(0, d_input - 1))
            else:
                return ('const', rng.uniform(-5, 5))
        
        # Internal node: binary or unary operator
        op = rng.choice(['+', '-', '*', '/', 'sin', 'cos', 'exp', 'log', 'abs', 'pow2'])
        
        if op in ['+', '-', '*', '/', 'pow2']:
            left = make_expression_tree(depth - 1, d_input, rng)
            right = make_expression_tree(depth - 1, d_input, rng)
            return (op, left, right)
        else:
            child = make_expression_tree(depth - 1, d_input, rng)
            return (op, child)
    
    def eval_expression(node, X):
        """Evaluate expression tree on input array X (normalized)."""
        op = node[0]
        if op == 'var':
            idx = node[1]
            return X[:, idx]
        elif op == 'const':
            return np.full(X.shape[0], node[1])
        elif op == '+':
            return eval_expression(node[1], X) + eval_expression(node[2], X)
        elif op == '-':
            return eval_expression(node[1], X) - eval_expression(node[2], X)
        elif op == '*':
            return eval_expression(node[1], X) * eval_expression(node[2], X)
        elif op == '/':
            a = eval_expression(node[1], X)
            b = eval_expression(node[2], X)
            b_safe = np.where(np.abs(b) < 1e-10, 1e-10 * np.sign(b + 1e-16), b)
            return a / b_safe
        elif op == 'sin':
            return np.sin(eval_expression(node[1], X))
        elif op == 'cos':
            return np.cos(eval_expression(node[1], X))
        elif op == 'exp':
            return np.exp(np.clip(eval_expression(node[1], X), -50, 50))
        elif op == 'log':
            a = eval_expression(node[1], X)
            a_safe = np.where(a < 1e-10, 1e-10, a)
            return np.log(a_safe)
        elif op == 'abs':
            return np.abs(eval_expression(node[1], X))
        elif op == 'pow2':
            a = eval_expression(node[1], X)
            return a * a
        else:
            return np.zeros(X.shape[0])
    
    def evolve_expressions(X_norm, y_norm, budget_time=60.0):
        """Evolve a population of expression trees."""
        rng = random.Random(42)
        pop_size = 50
        max_depth = 4
        population = [make_expression_tree(max_depth, d, rng) for _ in range(pop_size)]
        
        def fitness(node):
            try:
                pred = eval_expression(node, X_norm)
                if not np.all(np.isfinite(pred)):
                    return 1e10
                return np.mean((pred - y_norm) ** 2)
            except Exception:
                return 1e10
        
        # Initial evaluation
        scores = [fitness(ind) for ind in population]
        best_idx = np.argmin(scores)
        best_individual = population[best_idx]
        best_fitness = scores[best_idx]
        
        t0 = time.time()
        generation = 0
        while time.time() - t0 < budget_time:
            generation += 1
            # Selection: tournament
            def tournament_select():
                candidates = rng.sample(range(pop_size), 3)
                return min(candidates, key=lambda i: scores[i])
            
            new_population = []
            # Elitism: keep best
            new_population.append(copy.deepcopy(best_individual))
            
            for _ in range(pop_size - 1):
                # Crossover or mutation
                if rng.random() < 0.6:
                    # Subtree crossover
                    p1_idx = tournament_select()
                    p2_idx = tournament_select()
                    p1 = copy.deepcopy(population[p1_idx])
                    p2 = copy.deepcopy(population[p2_idx])
                    child = crossover(p1, p2, rng)
                else:
                    # Mutation
                    p1_idx = tournament_select()
                    child = mutate(copy.deepcopy(population[p1_idx]), rng, d)
                
                if child is not None:
                    new_population.append(child)
                else:
                    new_population.append(copy.deepcopy(population[tournament_select()]))
            
            population = new_population
            scores = [fitness(ind) for ind in population]
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_fitness:
                best_fitness = scores[best_idx]
                best_individual = copy.deepcopy(population[best_idx])
        
        return best_individual, best_fitness
    
    def get_subtree(node, rng):
        """Get a random subtree."""
        def collect_subtrees(n, path=()):
            results = [(n, path)]
            if n[0] in ['+', '-', '*', '/', 'pow2']:
                for child, idx in [(n[1], 1), (n[2], 2)]:
                    results.extend(collect_subtrees(child, path + (idx,)))
            elif n[0] in ['sin', 'cos', 'exp', 'log', 'abs']:
                results.extend(collect_subtrees(n[1], path + (1,)))
            return results
        
        all_subtrees = collect_subtrees(node)
        return rng.choice(all_subtrees)
    
    def replace_subtree(node, subtree, target_path, rng):
        """Replace subtree at given path."""
        if not target_path:
            return subtree
        idx = target_path[0]
        rest = target_path[1:]
        if node[0] in ['+', '-', '*', '/', 'pow2']:
            if idx == 1:
                new_left = replace_subtree(node[1], subtree, rest, rng)
                return (node[0], new_left, node[2])
            elif idx == 2:
                new_right = replace_subtree(node[2], subtree, rest, rng)
                return (node[0], node[1], new_right)
        elif node[0] in ['sin', 'cos', 'exp', 'log', 'abs']:
            if idx == 1:
                new_child = replace_subtree(node[1], subtree, rest, rng)
                return (node[0], new_child)
        return node
    
    def crossover(p1, p2, rng):
        """Subtree crossover."""
        try:
            subtree1, path1 = get_subtree(p1, rng)
            subtree2, path2 = get_subtree(p2, rng)
            child1 = replace_subtree(copy.deepcopy(p1), copy.deepcopy(subtree2), path1, rng)
            child2 = replace_subtree(copy.deepcopy(p2), copy.deepcopy(subtree1), path2, rng)
            return child1 if rng.random() < 0.5 else child2
        except Exception:
            return None
    
    def mutate(node, rng, d_input):
        """Mutate a subtree."""
        try:
            if rng.random() < 0.3:
                # Replace with new subtree
                return make_expression_tree(rng.randint(1, 3), d_input, rng)
            else:
                # Mutate a random node
                def random_walk(n):
                    if n[0] in ['+', '-', '*', '/', 'pow2']:
                        idx = rng.randint(1, 2)
                        n = list(n)
                        n[idx] = random_walk(n[idx])
                        return tuple(n)
                    elif n[0] in ['sin', 'cos', 'exp', 'log', 'abs']:
                        n = list(n)
                        n[1] = random_walk(n[1])
                        return tuple(n)
                    else:
                        # Leaf: change type or value
                        if rng.random() < 0.5:
                            return ('var', rng.randint(0, d_input - 1))
                        else:
                            return ('const', rng.uniform(-5, 5))
                return random_walk(node)
        except Exception:
            return None
    
    # Run GP if we have time
    gp_budget = min(45.0, deadline - time.time() - 30.0)
    if gp_budget > 5:
        try:
            best_tree, best_fit = evolve_expressions(X_norm, y_norm, gp_budget)
            if best_fit < 1e-6:
                def gp_predict(X):
                    X_n = (X - X_mean) / X_std
                    pred_norm = eval_expression(best_tree, X_n)
                    return denorm_predict(pred_norm)
                _try_update(gp_predict)
        except Exception:
            pass
    
    # --- Strategy 2: Neural network with gradient-free optimization ---
    def simple_nn(X_norm, weights, biases):
        """Forward pass through a simple MLP: input -> hidden (tanh) -> output."""
        h = np.tanh(X_norm @ weights[0] + biases[0])
        out = h @ weights[1] + biases[1]
        return out
    
    def nn_loss(params, X_norm, y_norm):
        w1, b1, w2, b2 = params
        try:
            pred = simple_nn(X_norm, [w1, w2], [b1, b2])
            if not np.all(np.isfinite(pred)):
                return 1e10
            return np.mean((pred - y_norm) ** 2)
        except Exception:
            return 1e10
    
    def optimize_nn(X_norm, y_norm, budget_time=40.0):
        """Optimize NN parameters using coordinate descent + random restarts."""
        hidden_size = max(4, min(16, n // 10))
        
        def init_params():
            w1 = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
            b1 = np.zeros(hidden_size)
            w2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
            b2 = np.zeros(1)
            return [w1, b1, w2, b2]
        
        best_params = None
        best_loss = np.inf
        
        t0 = time.time()
        n_restarts = 0
        while time.time() - t0 < budget_time:
            n_restarts += 1
            params = init_params()
            current_loss = nn_loss(params, X_norm, y_norm)
            
            # Coordinate descent with line search
            for step in range(500):
                if time.time() - t0 > budget_time:
                    break
                improved = False
                # Update each parameter block
                for i, p in enumerate(params):
                    original = p.copy()
                    # Try multiple step sizes
                    for lr in [0.1, 0.01, 0.001]:
                        perturbed = p + lr * np.random.randn(*p.shape)
                        test_params = params.copy()
                        test_params[i] = perturbed
                        new_loss = nn_loss(test_params, X_norm, y_norm)
                        if new_loss < current_loss:
                            params[i] = perturbed
                            current_loss = new_loss
                            improved = True
                            break
                if not improved:
                    # Try smaller perturbations
                    for i, p in enumerate(params):
                        perturbed = p + 0.001 * np.random.randn(*p.shape)
                        test_params = params.copy()
                        test_params[i] = perturbed
                        new_loss = nn_loss(test_params, X_norm, y_norm)
                        if new_loss < current_loss:
                            params[i] = perturbed
                            current_loss = new_loss
                            improved = True
                            break
                    if not improved:
                        break
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = [p.copy() for p in params]
        
        if best_params is None:
            return None
        return best_params, best_loss
    
    nn_budget = min(40.0, deadline - time.time() - 25.0)
    if nn_budget > 5:
        try:
            nn_result = optimize_nn(X_norm, y_norm, nn_budget)
            if nn_result is not None:
                params, loss = nn_result
                if loss < 1e-4:
                    def nn_predict(X):
                        X_n = (X - X_mean) / X_std
                        pred_norm = simple_nn(X_n, [params[0], params[2]], [params[1], params[3]])
                        return denorm_predict(pred_norm)
                    _try_update(nn_predict)
        except Exception:
            pass
    
    # --- Strategy 3: Polynomial regression (fast baseline) ---
    def make_poly_features(X, degree):
        cols = [np.ones(X.shape[0])]
        for total_deg in range(1, degree + 1):
            for indices in product(range(d), repeat=total_deg):
                if sorted(indices) != indices:
                    continue
                feat = np.ones(X.shape[0])
                for idx in indices:
                    feat = feat * X[:, idx]
                cols.append(feat)
        return np.column_stack(cols)
    
    from itertools import product
    for deg in [2, 3]:
        if time.time() > deadline - 15:
            break
        try:
            F = make_poly_features(X_train, deg)
            w, *_ = np.linalg.lstsq(F, y_train, rcond=None)
            _try_update(lambda X, w=w, deg=deg: make_poly_features(X, deg) @ w)
        except Exception:
            pass
    
    # --- Strategy 4: GP regression (scipy) as final fallback ---
    if time.time() < deadline - 10:
        try:
            from scipy.spatial.distance import cdist
            
            def gp_predict(X_pred, X_tr, y_tr, length_scale, sigma_f, noise):
                K_tr = sigma_f ** 2 * np.exp(-cdist(X_tr, X_tr) ** 2 / (2 * length_scale ** 2)) + noise * np.eye(len(X_tr))
                K_pred = sigma_f ** 2 * np.exp(-cdist(X_pred, X_tr) ** 2 / (2 * length_scale ** 2))
                try:
                    L = np.linalg.cholesky(K_tr)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr - y_mean))
                    mu = y_mean + K_pred @ alpha
                    return mu
                except Exception:
                    return np.full(X_pred.shape[0], y_mean)
            
            for ls in [0.5, 1.0, 2.0]:
                for sf in [1.0, 5.0]:
                    if time.time() > deadline - 5:
                        break
                    try:
                        _try_update(lambda X, l=ls, s=sf: gp_predict(X, X_train, y_train, l, s, 1e-8))
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final fallback: constant predictor
    if best_predictor is None:
        best_predictor = lambda X: np.full(X.shape[0], y_mean)
    
    return best_predictor
# EVOLVE-BLOCK-END
