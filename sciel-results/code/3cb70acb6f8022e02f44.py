"""Rewritten program: exploit residue-class structure + MILP polish for sum-free subsets."""
# EVOLVE-BLOCK-START
import time
import math
from random import Random

def construct():
    deadline = time.time() + 110.0  # safety margin before 120s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        """Check if a set is sum-free: no a+b=c for a,b,c in set (a=b allowed)."""
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def try_milp(values, timeout=20.0):
        """Try to find a larger sum-free subset using MILP via scipy.optimize.milp."""
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            import numpy as np
        except ImportError:
            return None

        n = len(values)
        if n > 300:
            return None  # too large for MILP in reasonable time

        # Build conflict graph: edge (i,j) means values[i] + values[j] is in the set
        val_to_idx = {v: i for i, v in enumerate(values)}
        conflicts = [set() for _ in range(n)]

        for i in range(n):
            for j in range(i, n):
                s = values[i] + values[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    # Conflict: can't have all three i, j, k
                    # This is a 3-uniform hyperedge, not a simple edge.
                    # For MILP, we need: x_i + x_j + x_k <= 2
                    pass

        # For small n, build the hypergraph constraints directly
        if n <= 80:
            c = np.ones(n)
            # Constraints: for each (i,j,k) with values[i]+values[j]==values[k], x_i+x_j+x_k <= 2
            A_list = []
            b_list = []
            for i in range(n):
                for j in range(i, n):
                    s = values[i] + values[j]
                    if s in val_to_idx:
                        k = val_to_idx[s]
                        row = np.zeros(n)
                        row[i] = 1
                        row[j] = 1
                        row[k] = 1
                        A_list.append(row)
                        b_list.append(2)

            if A_list:
                A = np.array(A_list)
                b = np.array(b_list)
            else:
                # No conflicts, take all
                return list(values)

            constraints = LinearConstraint(A, -np.inf, b)
            integrality = np.ones(n)
            bounds = Bounds(0, 1)

            result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                          options={'time_limit': timeout})

            if result.success and result.x is not None:
                selected = [values[i] for i in range(n) if result.x[i] > 0.5]
                if is_sum_free(selected):
                    return selected

        return None

    def greedy_prune(values):
        """Greedy: start with all, remove elements that create conflicts, prefer removing
        the one that participates in most conflicts."""
        s = set(values)
        # Build conflict count
        conflict_count = {v: 0 for v in s}
        for a in s:
            for b in s:
                c = a + b
                if c in s:
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1

        changed = True
        while changed:
            changed = False
            # Find element with max conflicts
            if not s:
                break
            max_conf = -1
            max_elem = None
            for v in s:
                # Recount conflicts for v
                cnt = 0
                for a in s:
                    for b in s:
                        if a + b == v:
                            cnt += 1
                    if v + a in s:
                        cnt += 1
                    if a + v in s:
                        cnt += 1
                if cnt > max_conf:
                    max_conf = cnt
                    max_elem = v
            if max_conf > 0 and max_elem is not None:
                s.remove(max_elem)
                changed = True

        return sorted(s)

    def residue_class_solution(values, mod):
        """For a given modulus, find the best residue class subset that is sum-free.
        A residue class r mod m is sum-free if 2r mod m != r mod m and no three elements
        in that class satisfy a+b=c. For a single residue class, this is always true
        if 2r != r mod m (i.e., r != 0 mod m/2 when m is even, etc.).
        More generally, we pick the largest set of residues R such that no r1+r2 in R (mod m)
        has an element in R. This is the max independent set in the sum graph on Z/mZ."""
        # Build sum graph on residues 0..mod-1
        # Edge between r1, r2 if (r1+r2) mod mod is also a valid residue we might pick
        # Actually: we want subset R of {0,...,mod-1} such that for all a,b in R, (a+b)%mod not in R
        # This is: R is sum-free in Z/modZ.
        
        # For small mod, brute force over all subsets
        if mod <= 12:
            best_size = 0
            best_residues = None
            for mask in range(1 << mod):
                residues = [i for i in range(mod) if (mask >> i) & 1]
                if not residues:
                    continue
                # Check sum-free
                valid = True
                for a in residues:
                    for b in residues:
                        if (a + b) % mod in residues:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Count elements in values with these residues
                    cnt = sum(1 for v in values if abs(v) % mod in residues)
                    # Actually need v % mod, but v can be negative
                    cnt = sum(1 for v in values if ((v % mod) + mod) % mod in residues)
                    if cnt > best_size:
                        best_size = cnt
                        best_residues = residues

            if best_residues is not None:
                subset = [v for v in values if ((v % mod) + mod) % mod in best_residues]
                if is_sum_free(subset):
                    return subset
                # If not sum-free (shouldn't happen), prune
                return greedy_prune(subset)

        return None

    def multi_residue_solution(values):
        """Try multiple moduli and pick the best residue-class-based solution."""
        best = []
        best_size = 0
        for mod in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]:
            result = residue_class_solution(values, mod)
            if result is not None and len(result) > best_size:
                best_size = len(result)
                best = result
        return best

    def local_search(values, max_iter=5000):
        """Local search: start from a good initial solution, try adding/removing/swapping."""
        # Start from multi-residue solution
        current = multi_residue_solution(values)
        if not current:
            current = [v for v in values if v % 2 != 0]  # odd numbers
        current_set = set(current)

        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try adding each element not in current
            for v in values:
                if v in current_set:
                    continue
                # Check if adding v keeps sum-free
                valid = True
                for a in current_set:
                    if a + v in current_set:
                        valid = False
                        break
                    if v + a in current_set:
                        valid = False
                        break
                    if 2 * v in current_set:
                        valid = False
                        break
                if valid:
                    current_set.add(v)
                    improved = True
                    break
            if not improved:
                # Try removing one element to allow adding two
                removed = None
                for v in list(current_set):
                    current_set.remove(v)
                    removed = v
                    # Try adding
                    added = False
                    for w in values:
                        if w in current_set or w == v:
                            continue
                        valid = True
                        for a in current_set:
                            if a + w in current_set:
                                valid = False
                                break
                        if valid:
                            current_set.add(w)
                            added = True
                            break
                    if not added:
                        current_set.add(v)  # restore
                        removed = None
                    break
                if removed is None:
                    break

        return sorted(current_set)

    # Process each instance
    results = []
    for size, seed in [(60, 801), (120, 802), (240, 803)]:
        values = instance(size, seed)

        # Get baseline from odd numbers
        odd_subset = [v for v in values if v % 2 != 0]
        best = odd_subset
        best_size = len(odd_subset)

        # Try multi-residue
        mr = multi_residue_solution(values)
        if mr and len(mr) > best_size:
            best = mr
            best_size = len(mr)

        # Try local search
        if time.time() < deadline - 30:
            ls = local_search(values)
            if len(ls) > best_size:
                best = ls
                best_size = len(ls)

        # Try MILP for small instances
        if size <= 120 and time.time() < deadline - 20:
            milp_result = try_milp(values, timeout=min(15.0, deadline - time.time() - 5))
            if milp_result and len(milp_result) > best_size:
                best = milp_result
                best_size = len(milp_result)

        results.append(best)

    return results
# EVOLVE-BLOCK-END
