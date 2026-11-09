# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 295.0  # 35s margin for 330s kill
    
    n, d = X_train.shape
    y_mean = np.mean(y_train)
    y_var = np.var(y_train)
    if y_var < 1e-12:
        return lambda X: np.full(len(X), y_mean)
    
    def check_time():
        return time.time() < deadline
    
    # Normalize features for numerical stability
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    # Normalize y
    y_std = np.std(y_train)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_train - y_mean) / y_std
    
    def safe_evaluate(model, X, y):
        try:
            preds = model(X)
            preds = np.asarray(preds, dtype=float)
            if preds.shape != y.shape:
                return np.inf
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return np.inf
            mse = np.mean((preds - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            return mse
        except Exception:
            return np.inf
    
    best_predict = None
    best_mse = np.inf
    
    def update_best(model):
        nonlocal best_predict, best_mse
        if model is None:
            return
        mse = safe_evaluate(model, X_train, y_train)
        if mse < best_mse:
            best_mse = mse
            best_predict = model
    
    # ============================================================
    # STRATEGY A: Genetic Programming (Symbolic Regression)
    # ============================================================
    # Terminals: x0, x1, ..., x{d-1}
    # Functions: +, -, *, /, sin, cos, exp, sqrt, abs, log
    
    class Node:
        __slots__ = ['type', 'value', 'left', 'right']
        def __init__(self, type, value=None, left=None, right=None):
            self.type = type
            self.value = value
            self.left = left
            self.right = right
        
        def eval(self, X):
            if self.type == 'terminal':
                return X[:, self.value]
            elif self.type == 'func':
                l = self.left.eval(X)
                r = self.right.eval(X) if self.right is not None else None
                if self.value == '+':
                    return l + r
                elif self.value == '-':
                    return l - r
                elif self.value == '*':
                    return l * r
                elif self.value == '/':
                    return np.divide(l, r, out=np.zeros_like(l), where=np.abs(r) > 1e-10)
                elif self.value == 'sin':
                    return np.sin(l)
                elif self.value == 'cos':
                    return np.cos(l)
                elif self.value == 'exp':
                    return np.exp(np.clip(l, -50, 50))
                elif self.value == 'sqrt':
                    return np.sqrt(np.maximum(l, 0))
                elif self.value == 'abs':
                    return np.abs(l)
                elif self.value == 'log':
                    return np.log(np.maximum(l, 1e-10))
                else:
                    return l
            return np.zeros(len(X))
        
        def clone(self):
            if self.type == 'terminal':
                return Node('terminal', self.value)
            else:
                l = self.left.clone() if self.left else None
                r = self.right.clone() if self.right else None
                return Node('func', self.value, l, r)
        
        def depth(self):
            if self.type == 'terminal':
                return 0
            ld = self.left.depth() if self.left else 0
            rd = self.right.depth() if self.right else 0
            return 1 + max(ld, rd)
        
        def size(self):
            if self.type == 'terminal':
                return 1
            ls = self.left.size() if self.left else 0
            rs = self.right.size() if self.right else 0
            return 1 + ls + rs
    
    unary_funcs = ['sin', 'cos', 'exp', 'sqrt', 'abs', 'log']
    binary_funcs = ['+', '-', '*', '/']
    
    def random_tree(max_depth, min_depth=0):
        if max_depth == 0 or (min_depth > 0 and random.random() < 0.3):
            return Node('terminal', random.randint(0, d - 1))
        if random.random() < 0.4:
            f = random.choice(unary_funcs)
            return Node('func', f, random_tree(max_depth - 1, min_depth - 1), None)
        else:
            f = random.choice(binary_funcs)
            return Node('func', f, 
                       random_tree(max_depth - 1, min_depth - 1),
                       random_tree(max_depth - 1, min_depth - 1))
    
    def mutate(tree, rate=0.3):
        new_tree = tree.clone()
        _mutate_recursive(new_tree, rate)
        return new_tree
    
    def _mutate_recursive(node, rate):
        if node.type == 'terminal':
            if random.random() < rate:
                node.value = random.randint(0, d - 1)
            return
        if random.random() < rate:
            if random.random() < 0.4:
                node.value = random.choice(unary_funcs)
                node.right = None
            else:
                node.value = random.choice(binary_funcs)
                if node.right is None:
                    node.right = random_tree(3)
        if node.left:
            _mutate_recursive(node.left, rate)
        if node.right:
            _mutate_recursive(node.right, rate)
    
    def crossover(t1, t2):
        t1 = t1.clone()
        t2 = t2.clone()
        # Random subtree swap
        def random_subtree(node):
            if node.type == 'terminal':
                return node
            if random.random() < 0.5:
                return random_subtree(node.left) if node.left else node
            elif node.right is not None and random.random() < 0.5:
                return random_subtree(node.right)
            else:
                return node
        s1 = random_subtree(t1)
        s2 = random_subtree(t2)
        s1, s2 = s2, s1
        return t1
    
    def gp_fitness(tree, X, y):
        try:
            pred = tree.eval(X)
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return np.inf
            mse = np.mean((pred - y) ** 2)
            if np.isnan(mse) or np.isinf(mse):
                return np.inf
            # Add complexity penalty
            penalty = 0.01 * tree.size() + 0.005 * tree.depth()
            return mse + penalty
        except Exception:
            return np.inf
    
    # Run GP
    pop_size = 60
    max_gen = 30
    max_depth_init = 4
    
    population = []
    for _ in range(pop_size):
        tree = random_tree(max_depth_init)
        population.append(tree)
    
    best_gp_tree = None
    best_gp_fit = np.inf
    
    for gen in range(max_gen):
        if not check_time():
            break
        
        # Evaluate fitness
        fits = [gp_fitness(t, X_train, y_norm) for t in population]
        
        # Track best
        min_idx = int(np.argmin(fits))
        if fits[min_idx] < best_gp_fit:
            best_gp_fit = fits[min_idx]
            best_gp_tree = population[min_idx].clone()
        
        # Create next generation
        new_pop = []
        
        # Elitism: keep top 5
        sorted_idx = np.argsort(fits)
        for i in range(min(5, len(sorted_idx))):
            new_pop.append(population[sorted_idx[i]].clone())
        
        while len(new_pop) < pop_size:
            if not check_time():
                break
            r = random.random()
            if r < 0.5:
                # Crossover
                p1 = population[int(np.random.choice(len(population), p=np.array([1/(f+1e-10) for f in fits])/sum(1/(f+1e-10) for f in fits)))]
                p2 = population[int(np.random.choice(len(population), p=np.array([1/(f+1e-10) for f in fits])/sum(1/(f+1e-10) for f in fits)))]
                child = crossover(p1, p2)
                if random.random() < 0.2:
                    child = mutate(child, 0.2)
                new_pop.append(child)
            elif r < 0.8:
                # Mutation
                p1 = population[int(np.random.choice(len(population), p=np.array([1/(f+1e-10) for f in fits])/sum(1/(f+1e-10) for f in fits)))]
                child = mutate(p1, 0.3)
                new_pop.append(child)
            else:
                # Random restart
                child = random_tree(max_depth_init)
                new_pop.append(child)
        
        population = new_pop
    
    # If GP found a good solution, use it
    if best_gp_tree is not None and best_gp_fit < 0.01:
        def predict_gp(X, tree, y_mean, y_std):
            try:
                pred = tree.eval(X)
                pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
                return y_mean + y_std * pred
            except Exception:
                return np.full(len(X), y_mean)
        update_best(lambda X: predict_gp(X, best_gp_tree, y_mean, y_std))
    
    # ============================================================
    # STRATEGY B: RBF with grid of centers (more robust)
    # ============================================================
    def make_rbf_model(gamma, n_centers_per_dim):
        # Create grid of centers
        if d == 1:
            centers = np.linspace(X_min[0], X_max[0], n_centers_per_dim).reshape(-1, 1)
        elif d == 2:
            x1_grid = np.linspace(X_min[0], X_max[0], n_centers_per_dim)
            x2_grid = np.linspace(X_min[1], X_max[1], n_centers_per_dim)
            X1, X2 = np.meshgrid(x1_grid, x2_grid)
            centers = np.column_stack([X1.ravel(), X2.ravel()])
        else:
            # For d>2, use random subset
            idx = np.random.choice(n, min(n_centers_per_dim ** 2, n), replace=False)
            centers = X_train[idx]
        
        n_c = len(centers)
        diffs = X_train[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        F = np.exp(-gamma * dists)
        
        lam = 1e-4
        A = F.T @ F + lam * np.eye(n_c + 1)
        A[0, 0] -= lam
        b = np.concatenate([np.ones(n_c), F.T @ y_norm])
        w = solve(A, b)
        
        def predict_rbf(X, centers, gamma, w, y_mean, y_std):
            diffs = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sum(diffs ** 2, axis=2)
            F_test = np.exp(-gamma * dists)
            F_test = np.column_stack([np.ones(len(F_test)), F_test])
            return y_mean + y_std * (F_test @ w)
        
        return lambda X: predict_rbf(X, centers, gamma, w, y_mean, y_std)
    
    X_range_mean = np.mean(X_range)
    base_gamma = 1.0 / (d * X_range_mean ** 2 + 1e-10)
    
    for g_mult in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        if not check_time():
            break
        for nc in [5, 8, 12]:
            if not check_time():
                break
            try:
                model = make_rbf_model(base_gamma * g_mult, nc)
                update_best(model)
            except Exception:
                pass
    
    # ============================================================
    # STRATEGY C: Polynomial with Chebyshev basis (better conditioned)
    # ============================================================
    def make_cheb_features(X, degree):
        Xn = (X - X_min) / X_range
        Xn = 2 * Xn - 1  # Map to [-1, 1]
        cols = [np.ones(len(Xn))]
        
        # Chebyshev polynomials of first kind
        def cheb_vals(x, deg):
            T0 = np.ones_like(x)
            if deg == 0:
                return [T0]
            T1 = x.copy()
            result = [T0, T1]
            for k in range(2, deg + 1):
                Tk = 2 * x * T1 - T0
                result.append(Tk)
                T0, T1 = T1, Tk
            return result
        
        cheb_sets = []
        for j in range(d):
            cheb_sets.append(cheb_vals(Xn[:, j], degree))
        
        # Add all combinations up to total degree
        for total_deg in range(1, degree + 1):
            for combo in _cheb_combinations(d, total_deg):
                term = np.ones(len(Xn))
                for j, k in combo:
                    term *= cheb_sets[j][k]
                cols.append(term)
        
        return np.column_stack(cols)
    
    def _cheb_combinations(d, total_deg):
        """Generate all combinations of (var, degree) that sum to total_deg"""
        if d == 1:
            return [(0, total_deg)]
        results = []
        for k in range(total_deg + 1):
            rest = _cheb_combinations(d - 1, total_deg - k)
            for r in rest:
                results.append((0, k) + r)
        return results
    
    for deg in [2, 3, 4, 5]:
        if not check_time():
            break
        try:
            F = make_cheb_features(X_train, deg)
            if F.shape[1] > 100:  # Limit complexity
                continue
            lam = 1e-4
            A = F.T @ F + lam * np.eye(F.shape[1])
            A[0, 0] -= lam
            w = solve(A, F.T @ y_norm)
            
            def predict_cheb(X, w, y_mean, y_std):
                F_test = make_cheb_features(X, deg)
                return y_mean + y_std * (F_test @ w)
            
            update_best(lambda X: predict_cheb(X, w, y_mean, y_std))
        except Exception:
            pass
    
    # ============================================================
    # STRATEGY D: Simple functional form search
    # ============================================================
    # Try all simple forms for each pair of variables
    simple_forms_2d = [
        lambda x1, x2: x1 + x2,
        lambda x1, x2: x1 - x2,
        lambda x1, x2: x2 - x1,
        lambda x1, x2: x1 * x2,
        lambda x1, x2: x1 / np.maximum(x2, 1e-10),
        lambda x1, x2: x2 / np.maximum(x1, 1e-10),
        lambda x1, x2: x1 ** 2 + x2 ** 2,
        lambda x1, x2: x1 ** 2 - x2 ** 2,
        lambda x1, x2: x2 ** 2 - x1 ** 2,
        lambda x1, x2: x1 ** 2 * x2,
        lambda x1, x2: x1 * x2 ** 2,
        lambda x1, x2: np.sin(x1) * np.cos(x2),
        lambda x1, x2: np.cos(x1) * np.sin(x2),
        lambda x1, x2: np.sin(x1 + x2),
        lambda x1, x2: np.cos(x1 + x2),
        lambda x1, x2: np.exp(-x1) * x2,
        lambda x1, x2: np.exp(-x2) * x1,
        lambda x1, x2: np.sqrt(np.maximum(x1, 0)) * x2,
        lambda x1, x2: x1 * np.sqrt(np.maximum(x2, 0)),
        lambda x1, x2: np.abs(x1 - x2),
        lambda x1, x2: np.maximum(x1, x2),
        lambda x1, x2: np.minimum(x1, x2),
        lambda x1, x2: (x1 + x2) ** 2,
        lambda x1, x2: (x1 - x2) ** 2,
        lambda x1, x2: x1 ** 3 + x2,
        lambda x1, x2: x1 + x2 ** 3,
        lambda x1, x2: x1 ** 2 * x2 ** 2,
        lambda x1, x2: np.sin(x1) + np.cos(x2),
        lambda x1, x2: x1 * np.sin(x2),
        lambda x1, x2: x2 * np.sin(x1),
    ]
    
    for i in range(d):
        for j in range(i + 1, d):
            if not check_time():
                break
            for form_fn in simple_forms_2d:
                if not check_time():
                    break
                try:
                    x1 = X_train[:, i]
                    x2 = X_train[:, j]
                    y_pred = form_fn(x1, x2)
                    # Linear fit: y = a * form + b
                    A_mat = np.column_stack([y_pred, np.ones(n)])
                    lam = 1e-6
                    A = A_mat.T @ A_mat + lam * np.eye(2)
                    A[1, 1] -= lam
                    w = solve(A, A_mat.T @ y_norm)
                    a, b = w[0], w[1]
                    mse = np.mean((a * y_pred + b - y_norm) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        def predict_form(X, i, j, form_fn, a, b, y_mean, y_std):
                            x1 = X[:, i]
                            x2 = X[:, j]
                            return y_mean + y_std * (a * form_fn(x1, x2) + b)
                        best_predict = lambda X, i=i, j=j, f=form_fn, a=a, b=b: predict_form(X, i, j, f, a, b, y_mean, y_std)
                except Exception:
                    pass
    
    # 1D simple forms
    simple_forms_1d = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x: x ** 3,
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: np.exp(x),
        lambda x: np.exp(-x),
        lambda x: np.sqrt(np.maximum(x, 0)),
        lambda x: np.abs(x),
        lambda x: np.log(np.maximum(x, 1e-10)),
        lambda x: 1.0 / np.maximum(x, 1e-10),
        lambda x: x * np.sin(x),
        lambda x: x * np.cos(x),
        lambda x: np.sin(x) / np.maximum(x, 1e-10),
        lambda x: x ** 2 * np.sin(x),
        lambda x: np.exp(-x ** 2 / 2),
    ]
    
    for i in range(d):
        if not check_time():
            break
        for form_fn in simple_forms_1d:
            if not check_time():
                break
            try:
                x1 = X_train[:, i]
                y_pred = form_fn(x1)
                A_mat = np.column_stack([y_pred, np.ones(n)])
                lam = 1e-6
                A = A_mat.T @ A_mat + lam * np.eye(2)
                A[1, 1] -= lam
                w = solve(A, A_mat.T @ y_norm)
                a, b = w[0], w[1]
                mse = np.mean((a * y_pred + b - y_norm) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    def predict_1d_form(X, i, form_fn, a, b, y_mean, y_std):
                        x1 = X[:, i]
                        return y_mean + y_std * (a * form_fn(x1) + b)
                    best_predict = lambda X, i=i, f=form_fn, a=a, b=b: predict_1d_form(X, i, f, a, b, y_mean, y_std)
            except Exception:
                pass
    
    # Fallback
    if best_predict is None:
        return lambda X: np.full(len(X), y_mean)
    
    return best_predict

# EVOLVE-BLOCK-END
