# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
from itertools import product
import random

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    deadline = time.time() + 620  # leave 10s margin before 630s kill

    def is_feasible(c, s):
        """Check if configuration (centers, sides) is feasible."""
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            cx, cy = c[i]
            si = s[i]
            if cx - si / 2 < -tol or cx + si / 2 > 1 + tol:
                return False
            if cy - si / 2 < -tol or cy + si / 2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2 - tol
                if abs(c[i][0] - c[j][0]) < t and abs(c[i][1] - c[j][1]) < t:
                    return False
        return True

    def objective(c, s):
        return np.sum(s)

    def solve_guillotine(root):
        """
        Solve LP for a given guillotine tree topology.
        root: a dictionary representing the guillotine tree.
        Variables: x[i], y[i], s[i] for i in 0..n-1
        Total variables: 3*n
        
        The guillotine tree structurally enforces non-overlap.
        We need to add constraints that enforce the tree structure:
        - For a vertical cut (left/right): all squares in left child must be to the left of all squares in right child.
          max_x_left + s_left/2 <= min_x_right - s_right/2  (approximately, but we need pairwise or aggregated)
          Actually, since they are in separate rectangular regions defined by the cut, 
          the constraint is:
          For any i in left, j in right: x[j] - x[i] >= (s[i] + s[j])/2
          This is O(N_left * N_right) constraints.
          
        - For a horizontal cut (top/bottom): all squares in top child must be above all squares in bottom child.
          For any i in top, j in bottom: y[i] - y[j] >= (s[i] + s[j])/2
          
        Additionally, we need boundary constraints for the leaves?
        No, the root covers [0,1]x[0,1].
        The leaves are just the squares.
        However, the "cut" position is not a variable in the standard LP formulation of squares.
        The cut position is implicitly determined by the positions of the squares.
        Wait, a guillotine cut splits the *region*.
        If we have a vertical cut, the left region is [0, w] x [0, 1] and right is [w, 1] x [0, 1].
        The squares in the left must fit in [0, w] x [0, 1].
        The squares in the right must fit in [w, 1] x [0, 1].
        
        This implies:
        For all i in left: x[i] + s[i]/2 <= w
        For all j in right: x[j] - s[j]/2 >= w
        
        Since w is not a variable, we can eliminate it:
        max_{i in left} (x[i] + s[i]/2) <= min_{j in right} (x[j] - s[j]/2)
        This is equivalent to:
        For all i in left, j in right: x[i] + s[i]/2 <= x[j] - s[j]/2
        => x[j] - x[i] >= (s[i] + s[j])/2
        
        This is exactly the same constraint as the "horizontal separation" in the flat topology!
        So, a guillotine tree is just a specific set of pairwise constraints.
        The advantage of the tree is that it guarantees the constraint graph is "acyclic" in a spatial sense, ensuring feasibility.
        
        So `solve_guillotine` is effectively `solve_topology` but with the topology derived from the tree.
        """
        # Extract topology from tree
        topology = {}
        
        def extract_constraints(node, left_set, right_set, top_set, bottom_set):
            # This is tricky because the sets of squares in subtrees need to be tracked.
            pass
            
        # Let's just build the topology dict directly from the tree structure.
        def get_squares(node):
            if 'square' in node:
                return [node['square']]
            else:
                return get_squares(node['left']) + get_squares(node['right'])

        def build_topology(node):
            if 'square' in node:
                return
            left_squares = get_squares(node['left'])
            right_squares = get_squares(node['right'])
            direction = node['direction'] # 'v' means vertical cut (left/right), 'h' means horizontal cut (top/bottom)
            
            # For all pairs (i, j) with i in left, j in right:
            for i in left_squares:
                for j in right_squares:
                    if direction == 'v':
                        # Left is to the left of Right
                        # x[j] - x[i] >= (s[i] + s[j])/2
                        # This corresponds to 'h' separation in my previous notation (horizontal axis separation)
                        # Wait, in my previous `solve_topology`:
                        # 'h' meant x[j] - x[i] >= ... (separated along x)
                        # 'v' meant y[j] - y[i] >= ... (separated along y)
                        # So if direction is 'v' (vertical cut), squares are separated along x.
                        # So we add 'h' to topology.
                        topology[(min(i, j), max(i, j))] = 'h'
                    else: # 'h' cut
                        # Top is above Bottom
                        # y[i] - y[j] >= ... where i in top, j in bottom
                        # Note: my previous `solve_topology` assumed i < j in the key (i,j) and enforced y[j] - y[i] >= ... for 'v'.
                        # This implies j is "higher" or "further" in the coordinate.
                        # Let's stick to the convention:
                        # Key (i, j) with i < j.
                        # If direction 'h' (horizontal cut): Top squares have higher y.
                        # If i is in Top and j is in Bottom: y[i] - y[j] >= ...
                        # If i is in Bottom and j is in Top: y[j] - y[i] >= ...
                        # So we need to know which one is "above".
                        # In a guillotine tree, 'left' is typically the first child.
                        # If direction is 'h', 'left' is Top, 'right' is Bottom.
                        # So for i in Top (left), j in Bottom (right): y[i] - y[j] >= ...
                        # This means y[j] - y[i] <= -(...).
                        # My `solve_topology` expects constraints of the form:
                        # 'h': x[j] - x[i] >= (s[i]+s[j])/2  (j is right of i)
                        # 'v': y[j] - y[i] >= (s[i]+s[j])/2  (j is above i)
                        
                        # So if i is Top and j is Bottom:
                        # We want y[i] >= y[j] + (s[i]+s[j])/2
                        # This is equivalent to y[j] - y[i] <= - (s[i]+s[j])/2
                        # This is NOT directly supported by the simple >= formulation in `solve_topology` unless we swap i and j or change signs.
                        # Let's check `solve_topology` again.
                        # It adds: row[i] = 1, row[j] = -1 for 'v'.
                        # Constraint: y[i] - y[j] + s[i]/2 + s[j]/2 <= 0  => y[i] + s[i]/2 <= y[j] - s[j]/2.
                        # This means i is BELOW j.
                        # So 'v' in `solve_topology` means "j is above i".
                        
                        # In the tree:
                        # If direction 'h': Left is Top, Right is Bottom.
                        # For i in Left (Top), j in Right (Bottom): i is above j.
                        # So we need "i is above j" constraint.
                        # `solve_topology` "v" constraint means "j is above i".
                        # So if we add topology[(i,j)] = 'v', it enforces j above i.
                        # But we want i above j.
                        # So we should add topology[(i,j)] = 'v' ONLY if we swap the roles?
                        # No, the key is (min, max).
                        # If i < j:
                        #   If i is Top and j is Bottom: We want i above j.
                        #   Constraint needed: y[i] - y[j] >= ...
                        #   `solve_topology` 'v' gives: y[j] - y[i] >= ... (j above i).
                        #   So we CANNOT use 'v' directly if i is Top and j is Bottom and i < j.
                        #   We would need a constraint y[i] - y[j] >= ...
                        #   This is the opposite sign.
                        
                        # Let's modify `solve_topology` to be more flexible or just build the LP directly in `solve_guillotine`.
                        pass
                        
            # Recurse
            build_topology(node['left'])
            build_topology(node['right'])
            
        # This is getting complicated. Let's write a dedicated LP builder for the tree.
        num_vars = 3 * n
        c_obj = np.zeros(num_vars)
        for i in range(n):
            c_obj[2*n + i] = -1.0
        
        A_ub_rows = []
        b_ub_rows = []
        
        # Boundary constraints
        for i in range(n):
            # x[i] - s[i]/2 >= 0
            row = np.zeros(num_vars)
            row[i] = -1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
            # x[i] + s[i]/2 <= 1
            row = np.zeros(num_vars)
            row[i] = 1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(1.0)
            
            # y[i] - s[i]/2 >= 0
            row = np.zeros(num_vars)
            row[n + i] = -1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
            
            # y[i] + s[i]/2 <= 1
            row = np.zeros(num_vars)
            row[n + i] = 1.0
            row[2*n + i] = 0.5
            A_ub_rows.append(row)
            b_ub_rows.append(1.0)
            
        def add_tree_constraints(node):
            if 'square' in node:
                return
            
            left_squares = get_squares(node['left'])
            right_squares = get_squares(node['right'])
            
            if node['direction'] == 'v':
                # Vertical cut: Left is left of Right.
                # For i in Left, j in Right:
                # x[j] - x[i] >= (s[i] + s[j])/2
                # x[i] - x[j] + s[i]/2 + s[j]/2 <= 0
                for i in left_squares:
                    for j in right_squares:
                        row = np.zeros(num_vars)
                        row[i] = 1.0       # x[i]
                        row[j] = -1.0      # x[j]
                        row[2*n + i] = 0.5 # s[i]/2
                        row[2*n + j] = 0.5 # s[j]/2
                        A_ub_rows.append(row)
                        b_ub_rows.append(0.0)
            else:
                # Horizontal cut: Left is Top, Right is Bottom.
                # For i in Left (Top), j in Right (Bottom):
                # y[i] - y[j] >= (s[i] + s[j])/2
                # y[j] - y[i] + s[i]/2 + s[j]/2 <= 0
                for i in left_squares:
                    for j in right_squares:
                        row = np.zeros(num_vars)
                        row[n + i] = -1.0  # -y[i]  (Wait, y[i] - y[j] >= ... => -y[i] + y[j] <= -... )
                        # Let's re-derive:
                        # y[i] - y[j] >= s[i]/2 + s[j]/2
                        # -y[i] + y[j] <= -s[i]/2 - s[j]/2
                        # This doesn't fit the standard form A_ub x <= b_ub with positive coefficients easily if we want to reuse the previous logic.
                        # Previous logic for 'v' (j above i) was:
                        # y[j] - y[i] >= s[i]/2 + s[j]/2
                        # -y[j] + y[i] + s[i]/2 + s[j]/2 <= 0
                        # Here we have i above j.
                        # So it's symmetric.
                        # Constraint: y[i] - y[j] >= s[i]/2 + s[j]/2
                        # => -y[i] + y[j] + s[i]/2 + s[j]/2 <= 0 ? No.
                        # => y[j] - y[i] <= - (s[i]/2 + s[j]/2)
                        # This is a "<=" constraint with a negative RHS.
                        # Linprog handles A_ub x <= b_ub.
                        # Row: -y[i] + y[j] ... 
                        # Let's just use the standard form:
                        # We want to enforce that the bottom of i is above the top of j.
                        # y[i] - s[i]/2 >= y[j] + s[j]/2
                        # y[i] - y[j] - s[i]/2 - s[j]/2 >= 0
                        # -y[i] + y[j] + s[i]/2 + s[j]/2 <= 0
                        row = np.zeros(num_vars)
                        row[n + i] = -1.0  # -y[i]
                        row[n + j] = 1.0   # y[j]
                        row[2*n + i] = 0.5 # s[i]/2
                        row[2*n + j] = 0.5 # s[j]/2
                        A_ub_rows.append(row)
                        b_ub_rows.append(0.0)
                        
            add_tree_constraints(node['left'])
            add_tree_constraints(node['right'])
            
        add_tree_constraints(root)
        
        if A_ub_rows:
            A_ub = np.array(A_ub_rows)
            b_ub = np.array(b_ub_rows)
        else:
            A_ub = None
            b_ub = None
            
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((0.0, 1.0))
        for i in range(n):
            bounds.append((1e-9, 1.0))
            
        try:
            result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if result.status == 0:
                x_opt = result.x
                c = np.zeros((n, 2))
                s = np.zeros(n)
                for i in range(n):
                    c[i, 0] = x_opt[i]
                    c[i, 1] = x_opt[n + i]
                    s[i] = x_opt[2*n + i]
                return c, s, result.fun * -1
            else:
                return None, None, -1
        except Exception:
            return None, None, -1

    def generate_guillotine_topologies():
        """
        Generate random guillotine trees.
        """
        topologies = []
        
        # Deterministic good structures
        # 1. Full binary tree (balanced)
        def make_balanced_tree(squares):
            if len(squares) == 1:
                return {'square': squares[0]}
            mid = len(squares) // 2
            left = squares[:mid]
            right = squares[mid:]
            direction = 'v' if len(left) <= len(right) else 'h' # Arbitrary
            # Actually, direction doesn't matter for the structure, just the split.
            # But for variety, let's alternate or randomize.
            return {
                'left': make_balanced_tree(left),
                'right': make_balanced_tree(right),
                'direction': 'v'
            }
            
        topologies.append(make_balanced_tree(list(range(n))))
        
        # 2. Combs (one big, rest small)
        # Square 0 is big, rest are in a strip
        def make_comb_tree(squares, big_first=True):
            if len(squares) == 1:
                return {'square': squares[0]}
            if big_first:
                big = squares[0]
                smalls = squares[1:]
                return {
                    'left': {'square': big},
                    'right': make_comb_tree(smalls, big_first=False),
                    'direction': 'v'
                }
            else:
                big = squares[-1]
                smalls = squares[:-1]
                return {
                    'left': make_comb_tree(smalls, big_first=False),
                    'right': {'square': big},
                    'direction': 'v'
                }
                
        topologies.append(make_comb_tree(list(range(n)), True))
        topologies.append(make_comb_tree(list(range(n)), False))
        
        # Random trees
        np.random.seed(42)
        random.seed(42)
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Randomly assign squares to a tree
            # We can build the tree by repeatedly picking a random subset for left and right?
            # No, a binary tree on N leaves can be generated by random recursive splits.
            
            def make_random_tree(squares):
                if len(squares) == 1:
                    return {'square': squares[0]}
                # Pick a random split point
                k = random.randint(1, len(squares) - 1)
                # Shuffle squares to randomize which ones go left/right
                shuffled = squares.copy()
                random.shuffle(shuffled)
                left = shuffled[:k]
                right = shuffled[k:]
                direction = 'v' if random.random() < 0.5 else 'h'
                return {
                    'left': make_random_tree(left),
                    'right': make_random_tree(right),
                    'direction': direction
                }
                
            tree = make_random_tree(list(range(n)))
            topologies.append(tree)
            
        return topologies

    def refine_with_local_search(c, s, max_rounds=100):
        """
        Local search to improve side lengths.
        """
        c = c.copy()
        s = s.copy()
        best_val = objective(c, s)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            order = np.random.permutation(n)
            for i in order:
                old_c = c[i].copy()
                old_s = s[i]
                
                # Try to increase side length
                max_s = min(2*old_c[0], 2*(1-old_c[0]), 2*old_c[1], 2*(1-old_c[1])) + 2*tol
                max_s = min(max_s, 1.0)
                
                if max_s <= old_s + 1e-12:
                    continue
                
                best_s = old_s
                best_c = old_c.copy()
                
                lo, hi = old_s, max_s
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if mid <= best_s + 1e-13:
                        break
                    
                    found = False
                    # Try current position
                    ok = True
                    for j in range(n):
                        if j == i:
                            continue
                        t = (mid + s[j]) / 2 - tol
                        if abs(old_c[0] - c[j][0]) < t and abs(old_c[1] - c[j][1]) < t:
                            ok = False
                            break
                    if ok and old_c[0] - mid/2 >= -tol and old_c[0] + mid/2 <= 1+tol \
                            and old_c[1] - mid/2 >= -tol and old_c[1] + mid/2 <= 1+tol:
                        found = True
                        best_c = old_c.copy()
                    
                    if not found:
                        # Try random positions
                        for _ in range(50):
                            nx = np.random.uniform(mid/2, 1-mid/2)
                            ny = np.random.uniform(mid/2, 1-mid/2)
                            ok = True
                            for j in range(n):
                                if j == i:
                                    continue
                                t = (mid + s[j]) / 2 - tol
                                if abs(nx - c[j][0]) < t and abs(ny - c[j][1]) < t:
                                    ok = False
                                    break
                            if ok:
                                found = True
                                best_c = np.array([nx, ny])
                                break
                    
                    if found:
                        best_s = mid
                        lo = mid
                    else:
                        hi = mid
                
                if best_s > s[i] + 1e-10:
                    s[i] = best_s
                    c[i] = best_c
                    improved = True
            
            if not improved:
                break
        
        return c, s

    # Main optimization
    best_c = None
    best_s = None
    best_val = -1
    
    # Generate guillotine topologies
    topologies = generate_guillotine_topologies()
    
    # Solve LP for each topology
    for idx, tree in enumerate(topologies):
        if time.time() > deadline:
            break
        
        c, s, val = solve_guillotine(tree)
        if c is not None and is_feasible(c, s):
            val = objective(c, s)
            if val > best_val:
                best_val = val
                best_c = c.copy()
                best_s = s.copy()
                
                # Try local refinement
                c_ref, s_ref = refine_with_local_search(c, s, max_rounds=50)
                if is_feasible(c_ref, s_ref):
                    val_ref = objective(c_ref, s_ref)
                    if val_ref > best_val:
                        best_val = val_ref
                        best_c = c_ref.copy()
                        best_s = s_ref.copy()
    
    # If no topology worked, use fallback
    if best_c is None:
        centers = np.array([
            [0.25, 0.25], [0.75, 0.25],
            [0.125, 0.875], [0.375, 0.875], [0.625, 0.875], [0.875, 0.875],
            [0.125, 0.625], [0.375, 0.625], [0.625, 0.625], [0.875, 0.625],
        ], dtype=np.float64)
        sides = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25], dtype=np.float64)
        best_c = centers
        best_s = sides
    
    # Final local search polish
    c_final, s_final = refine_with_local_search(best_c, best_s, max_rounds=200)
    if is_feasible(c_final, s_final):
        val_final = objective(c_final, s_final)
        if val_final > best_val:
            best_val = val_final
            best_c = c_final.copy()
            best_s = s_final.copy()
    
    best_s = np.maximum(best_s, 1e-9)
    
    return best_c, best_s

# EVOLVE-BLOCK-END
