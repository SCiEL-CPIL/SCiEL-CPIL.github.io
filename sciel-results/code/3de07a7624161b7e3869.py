"""Combined program: SAT-based checker + multi-start random search + greedy pruning
for a 5-uniform non-2-colourable hypergraph (minimising edges)."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """Return a 5-uniform hypergraph with no proper 2-colouring, minimising edges.

    Strategy:
      1. Fast SAT-based check (unit propagation + backtracking) for non-2-colourability.
      2. Multi-start: try several n (9..12) and several random initial edge sets.
      3. Greedy pruning: repeatedly try removing an edge; keep if still non-2-colourable.
      4. Keep the best (fewest edges) found within the time budget.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # ---------- Fast SAT solver for 2-colourability ----------
    # A hypergraph is 2-colourable iff there exists a 2-colouring (assignment of
    # True/False to vertices) such that every edge has both colours, i.e., no edge
    # is monochromatic.  We want to know if NO such colouring exists.
    #
    # Equivalent: is the CNF "for every edge, not(all same)" satisfiable?
    # For edge e = {v1..v5}, "not all same" means: not(all True) AND not(all False).
    #   not(all True):  (¬v1 ∨ ¬v2 ∨ ¬v3 ∨ ¬v4 ∨ ¬v5)
    #   not(all False): (v1 ∨ v2 ∨ v3 ∨ v4 ∨ v5)
    # So each edge gives two clauses. The hypergraph is 2-colourable iff this CNF
    # is satisfiable. Non-2-colourable iff the CNF is UNSAT.
    #
    # We implement a DPLL with unit propagation.

    def is_non_2colorable(edges, n):
        """Return True if the hypergraph has NO proper 2-colouring (i.e., is not 2-colourable)."""
        if not edges:
            return False  # empty hypergraph is trivially 2-colourable

        # Build clauses: for each edge, two clauses (OR of all, OR of all negated)
        # Represent variables as 0..n-1, clause as list of (var, sign) where sign=1 means var, -1 means ¬var
        clauses = []
        for e in edges:
            # Clause 1: v1 ∨ v2 ∨ ... ∨ v5  (not all False)
            clauses.append([(v, 1) for v in e])
            # Clause 2: ¬v1 ∨ ¬v2 ∨ ... ∨ ¬v5  (not all True)
            clauses.append([(v, -1) for v in e])

        # DPLL with unit propagation
        def dpll(clauses, assignment):
            """assignment: dict var -> True/False. Returns True if satisfiable."""
            # Unit propagation
            changed = True
            while changed:
                changed = False
                new_clauses = []
                for cl in clauses:
                    satisfied = False
                    new_cl = []
                    for (v, s) in cl:
                        if v in assignment:
                            if s * (1 if assignment[v] else -1) == 1:
                                satisfied = True
                                break
                        else:
                            new_cl.append((v, s))
                    if not satisfied:
                        if not new_cl:
                            return False  # empty clause: contradiction
                        if len(new_cl) == 1:
                            # Unit clause: assign
                            v, s = new_cl[0]
                            assignment[v] = (s == 1)
                            changed = True
                        else:
                            new_clauses.append(new_cl)
                    # else: clause satisfied, drop it
                clauses = new_clauses

            if not clauses:
                return True  # all clauses satisfied

            # Pick a variable (first unassigned in the first clause)
            v = clauses[0][0][0]
            # Try True
            assignment[v] = True
            if dpll([list(c) for c in clauses], dict(assignment)):
                return True
            del assignment[v]
            # Try False
            assignment[v] = False
            if dpll([list(c) for c in clauses], dict(assignment)):
                return True
            del assignment[v]
            return False

        # Quick pre-check: if n <= 12, we can also use bitmask enumeration as a cross-check
        # but DPLL should be fast enough. Let's use DPLL.
        # Actually, for very small n, bitmask is simpler and faster in Python.
        if n <= 10:
            # Bitmask enumeration
            edge_masks = []
            for e in edges:
                m = 0
                for v in e:
                    m |= (1 << v)
                edge_masks.append(m)
            for mask in range(1 << n):
                valid = True
                for em in edge_masks:
                    c = bin(em & mask).count('1')
                    if c == 0 or c == 5:
                        valid = False
                        break
                if valid:
                    return False
            return True
        else:
            # Use DPLL
            return not dpll([list(c) for c in clauses], {})

    # ---------- Greedy pruning ----------
    def prune_edges(edges, n, time_limit):
        """Try to remove edges while preserving non-2-colourability."""
        current = list(edges)
        random.shuffle(current)
        i = 0
        while i < len(current):
            if time.time() > time_limit:
                break
            candidate = current[:i] + current[i+1:]
            if is_non_2colorable(candidate, n):
                current = candidate
                # Do not increment i
            else:
                i += 1
        return current

    # ---------- Multi-start search ----------
    best_edges = None
    best_n = 9

    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 5.0:
            break

        all_possible = list(combinations(range(n), 5))
        total = len(all_possible)

        # Try several random initial sizes
        for attempt in range(50):
            if time.time() > deadline - 3.0:
                break

            # Start with a moderately dense set
            k_init = random.randint(40, min(70, total))
            random.shuffle(all_possible)
            initial = [list(e) for e in all_possible[:k_init]]

            if not is_non_2colorable(initial, n):
                continue

            # Prune
            prune_deadline = min(time.time() + 1.5, deadline - 2.0)
            pruned = prune_edges(initial, n, prune_deadline)

            if best_edges is None or len(pruned) < len(best_edges):
                best_edges = pruned
                best_n = n

                # If we've found a very good solution, we can keep searching
                # but we might also stop early if we hit a known lower bound
                if len(pruned) <= 35:
                    # Keep searching for a bit, but don't waste too much time
                    if time.time() > deadline - 10.0:
                        break

        if time.time() > deadline - 10.0:
            break

    # Fallback: complete 5-graph on 9 vertices
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
