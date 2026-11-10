"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
sys.setrecursionlimit(100000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

N = P['n']
VSET = frozenset(range(N))

# Build adjacency as sets, and edge->id map
_adj = [set() for _ in range(N)]
_edge_id = {}
_eid = 0
for (a, b) in P['edges']:
    _adj[a].add(b)
    _adj[b].add(a)
    key = (a, b) if a < b else (b, a)
    _edge_id[key] = _eid
    _eid += 1
M = _eid

# Precompute neighbor lists (sorted) for deterministic iteration
_nei = [sorted(_adj[i]) for i in range(N)]


def _edge_key(a, b):
    return (a, b) if a < b else (b, a)


class _Solver:
    def __init__(self, deadline):
        self.deadline = deadline
        self.used = [False] * M  # edge usage
        self.cycles = []
        self.best = []
        self.nodes = 0
        self._stop = False

    def _elapsed(self):
        return time.time() - self.deadline[0]

    def _check_deadline(self):
        if self._elapsed() > 55.0:
            self._stop = True
            return True
        return False

    def _update_best(self):
        if len(self.cycles) > len(self.best):
            self.best = [list(c) for c in self.cycles]

    def _find_cycle(self, start, forbidden_edges):
        """
        Find a Hamiltonian cycle (using all N vertices) starting/ending at `start`,
        avoiding edges in forbidden_edges (a set of edge-ids already used) plus
        the currently used edges (self.used).
        Returns list of vertices [v0,...,v_{N-1}] with v0=start, or None.
        """
        # We build a path of length N-1 starting at start, then need edge back to start.
        path = [start]
        visited = [False] * N
        visited[start] = True

        # Precompute for start: neighbors that have an edge to start (to close cycle)
        # and are not forbidden.

        def dfs(current, depth):
            if self._stop:
                return None
            self.nodes += 1
            if (self.nodes & 255) == 0 and self._check_deadline():
                return None
            if depth == N - 1:
                # Need edge current -> start
                ek = _edge_key(current, start)
                eid = _edge_id.get(ek)
                if eid is None:
                    return None
                if self.used[eid] or eid in forbidden_edges:
                    return None
                return list(path)
            # Try neighbors
            for nb in _nei[current]:
                if visited[nb]:
                    continue
                ek = _edge_key(current, nb)
                eid = _edge_id.get(ek)
                if eid is None:
                    continue
                if self.used[eid] or eid in forbidden_edges:
                    continue
                # Prune: ensure we can still reach start from nb eventually (simple degree check)
                path.append(nb)
                visited[nb] = True
                res = dfs(nb, depth + 1)
                if res is not None:
                    return res
                path.pop()
                visited[nb] = False
            return None

        return dfs(start, 0)

    def solve(self):
        # Greedy + backtracking: repeatedly find a Hamiltonian cycle, mark its edges used.
        # Use backtracking over the choice of cycle at each level.
        self._recursive(0)
        return self.best

    def _recursive(self, depth):
        if self._stop:
            return
        self._update_best()
        if len(self.cycles) >= 10:
            return
        # Try to find a cycle; iterate over possible starting vertices and
        # use different neighbor orderings to get diversity.
        # Strategy: try each start vertex 0..N-1; for each, run _find_cycle.
        # To get multiple options, we can try with different "forbidden" perturbations.
        # Simpler: try starts in a fixed order; first success commits.
        # For backtracking diversity, try all starts and pick first that works;
        # but to allow alternative cycles, we need to enumerate. We'll do a limited
        # enumeration: for the current depth, try each start vertex, and for each,
        # the _find_cycle returns one cycle (the first found). We commit to the first
        # start that yields a cycle. This is greedy. To add backtracking, we'd need
        # to try alternatives when dead end; but greedy Hamiltonian packing is strong.
        # We'll implement greedy with a few restart strategies if stuck.
        for start in range(N):
            if self._stop:
                return
            cyc = self._find_cycle(start, set())
            if cyc is None:
                continue
            # Commit this cycle
            eids = []
            ok = True
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                eid = _edge_id.get(_edge_key(a, b))
                if eid is None or self.used[eid]:
                    ok = False
                    break
                eids.append(eid)
            if not ok:
                continue
            for eid in eids:
                self.used[eid] = True
            self.cycles.append(cyc)
            self._recursive(depth + 1)
            # Undo
            for eid in eids:
                self.used[eid] = False
            self.cycles.pop()
            # After backtracking, continue trying other starts for this depth
            # (so we explore alternative cycles)
            if self._stop:
                return
            # Limit branching: only try a few alternatives to control explosion
            # We'll break after the first successful commitment to keep it fast,
            # but allow a couple of alternatives.
            # For now, break to keep greedy; the outer restarts handle diversity.
            break


def construct():
    deadline = [time.time()]
    best = []
    # Multiple restarts with different start orders / neighbor orderings
    import random
    rng = random.Random(12345)

    strategies = []
    # Strategy 0: natural order
    strategies.append((list(range(N)), 'natural'))
    # Strategy 1: reverse
    strategies.append((list(range(N - 1, -1, -1)), 'reverse'))
    # Strategy 2: shuffled
    for _ in range(4):
        s = list(range(N))
        rng.shuffle(s)
        strategies.append((s, 'shuffle'))

    for start_order, name in strategies:
        if time.time() - deadline[0] > 55.0:
            break
        # For each start order, we temporarily reorder _nei? That's global.
        # Instead, we pass a custom neighbor ordering by monkey-patching is messy.
        # We'll just rely on start order diversity; neighbor order stays natural.
        solver = _Solver(deadline)
        # Override the start iteration order by wrapping
        orig_recursive = solver._recursive
        # We'll just run solve but with a custom start list by modifying the loop.
        # Simplest: call a variant.
        res = solver.solve_with_order(start_order)
        if len(res) > len(best):
            best = res
        if solver._stop:
            break

    # Ensure types: list of lists of Python ints
    out = []
    for c in best:
        out.append([int(v) for v in c])
    if not out:
        # Fallback: return a single Hamiltonian cycle if we found any; else a trivial valid cycle
        # Find any Hamiltonian cycle quickly
        solver = _Solver(deadline)
        cyc = solver._find_cycle(0, set())
        if cyc is not None:
            out = [[int(v) for v in cyc]]
        else:
            # Last resort: find any simple cycle of length >=3 (shouldn't happen)
            # Use a small cycle
            # Find a triangle
            for a in range(N):
                for b in _nei[a]:
                    if b <= a:
                        continue
                    for c in _nei[b]:
                        if c == a or c == b:
                            continue
                        if c in _adj[a]:
                            out = [[int(a), int(b), int(c)]]
                            break
                    if out:
                        break
                if out:
                    break
    return out


# Add the solve_with_order method to _Solver
def _solve_with_order(self, start_order):
    self._update_best()
    if len(self.cycles) >= 10 or self._stop:
        return self.best
    for start in start_order:
        if self._stop:
            break
        cyc = self._find_cycle(start, set())
        if cyc is None:
            continue
        eids = []
        ok = True
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            eid = _edge_id.get(_edge_key(a, b))
            if eid is None or self.used[eid]:
                ok = False
                break
            eids.append(eid)
        if not ok:
            continue
        for eid in eids:
            self.used[eid] = True
        self.cycles.append(cyc)
        self._recursive_with_order(len(self.cycles) - 1, start_order)
        for eid in eids:
            self.used[eid] = False
        self.cycles.pop()
        if self._stop:
            break
        # Continue trying other starts for diversity
    return self.best


def _recursive_with_order(self, depth, start_order):
    if self._stop:
        return
    self._update_best()
    if len(self.cycles) >= 10:
        return
    for start in start_order:
        if self._stop:
            return
        cyc = self._find_cycle(start, set())
        if cyc is None:
            continue
        eids = []
        ok = True
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            eid = _edge_id.get(_edge_key(a, b))
            if eid is None or self.used[eid]:
                ok = False
                break
            eids.append(eid)
        if not ok:
            continue
        for eid in eids:
            self.used[eid] = True
        self.cycles.append(cyc)
        self._recursive_with_order(depth + 1, start_order)
        for eid in eids:
            self.used[eid] = False
        self.cycles.pop()
        if self._stop:
            return
        break


_Solver.solve_with_order = _solve_with_order
_Solver._recursive_with_order = _recursive_with_order
# EVOLVE-BLOCK-END
