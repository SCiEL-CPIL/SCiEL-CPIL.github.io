"""Dense odd-order graphs avoiding equal-degree path endpoints.

Strategy: fast bitmask validity check + greedy construction + local search
with random restarts and perturbation, under an explicit wall-clock deadline.
"""
import time
import random

N = 15
FULL = (1 << N) - 1
DEADLINE = time.time() + 85.0  # safety margin before the 90s kill
RNG = random.Random(1234567)


def _is_valid(adj, deg):
    # For every edge (u,v), a 3-path s-u-v-t (s!=t) is valid iff deg[s] != deg[t].
    # The edge is OK iff among neighbors of u (excluding v) and neighbors of v
    # (excluding u), NO pair has equal degree.
    for u in range(N):
        nu = adj[u]
        for v in range(u + 1, N):
            if not (nu >> v) & 1:
                continue
            du = deg[u]
            dv = deg[v]
            su = nu & ~(1 << v)
            sv = adj[v] & ~(1 << u)
            # collect degrees of s in su and t in sv
            # quick rejection: if there is s in su with deg==dv -> bad (t=v? no, t in sv)
            # actually need s in su, t in sv with deg[s]==deg[t]
            ds = []
            x = su
            while x:
                lsb = x & (-x)
                s = lsb.bit_length() - 1
                ds.append(deg[s])
                x ^= lsb
            t = sv
            while t:
                lsb = t & (-t)
                ti = lsb.bit_length() - 1
                dt = deg[ti]
                if dt in ds:
                    return False
                t ^= lsb
    return True


def _deg_from_adj(adj):
    return [bin(adj[i]).count("1") for i in range(N)]


def _edges_from_adj(adj):
    out = []
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] >> v) & 1:
                out.append([u, v])
    return out


def _greedy(adj, deg, remaining, deadline):
    # repeatedly add a random feasible edge until none feasible
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # build candidate list of missing edges
        cands = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    cands.append((u, v))
        RNG.shuffle(cands)
        for (u, v) in cands:
            if time.time() >= deadline:
                break
            # check feasibility of adding u-v
            du = deg[u]
            dv = deg[v]
            su = adj[u] & ~(1 << v)
            sv = adj[v] & ~(1 << u)
            ds = []
            x = su
            while x:
                lsb = x & (-x)
                s = lsb.bit_length() - 1
                ds.append(deg[s])
                x ^= lsb
            ok = True
            t = sv
            while t:
                lsb = t & (-t)
                ti = lsb.bit_length() - 1
                if deg[ti] in ds:
                    ok = False
                    break
                t ^= lsb
            if not ok:
                continue
            # also must ensure new edge itself doesn't create bad 3-path:
            # with u-v present, endpoints s in su, t in sv must have deg[s]!=deg[t]
            # (that's the same check above) AND deg[u] vs deg[? ] no: endpoints are s and t
            # but also paths where one endpoint is u? s-u-v-t: s and t are the endpoints.
            # s must be neighbor of u (not v), t neighbor of v (not u). Covered.
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            remaining -= 1
            changed = True
    return adj, deg, remaining


def _local_search(adj, deg, deadline):
    # try adding random missing edges; if invalid, try removing random present edges
    best_edges = sum(deg) // 2
    best_adj = [a for a in adj]
    best_deg = [d for d in deg]
    stall = 0
    while time.time() < deadline:
        # try add
        improved = False
        # pick random missing edge
        for _ in range(60):
            if time.time() >= deadline:
                break
            u = RNG.randrange(N)
            v = RNG.randrange(N)
            if u == v or ((adj[u] >> v) & 1):
                continue
            du = deg[u]; dv = deg[v]
            su = adj[u] & ~(1 << v)
            sv = adj[v] & ~(1 << u)
            ds = []
            x = su
            while x:
                lsb = x & (-x)
                s = lsb.bit_length() - 1
                ds.append(deg[s])
                x ^= lsb
            ok = True
            t = sv
            while t:
                lsb = t & (-t)
                ti = lsb.bit_length() - 1
                if deg[ti] in ds:
                    ok = False
                    break
                t ^= lsb
            if ok:
                adj[u] |= (1 << v); adj[v] |= (1 << u)
                deg[u] += 1; deg[v] += 1
                if _is_valid(adj, deg):
                    best_edges = sum(deg) // 2
                    best_adj = [a for a in adj]
                    best_deg = [d for d in deg]
                    improved = True
                    stall = 0
                    break
                else:
                    adj[u] &= ~(1 << v); adj[v] &= ~(1 << u)
                    deg[u] -= 1; deg[v] -= 1
        if improved:
            continue
        # try remove (perturbation)
        for _ in range(30):
            if time.time() >= deadline:
                break
            u = RNG.randrange(N)
            v = RNG.randrange(N)
            if u == v or not ((adj[u] >> v) & 1):
                continue
            adj[u] &= ~(1 << v); adj[v] &= ~(1 << u)
            deg[u] -= 1; deg[v] -= 1
            stall += 1
            break
        if stall > 40:
            # reset to best and restart perturbation harder
            adj = [a for a in best_adj]
            deg = [d for d in best_deg]
            stall = 0
    return best_adj, best_deg, best_edges


def construct():
    start = time.time()
    deadline = start + 85.0
    best_edges = 0
    best_adj = [0] * N
    best_deg = [0] * N

    while time.time() < deadline:
        # fresh start: empty graph
        adj = [0] * N
        deg = [0] * N
        remaining = N * (N - 1) // 2
        adj, deg, remaining = _greedy(adj, deg, remaining, deadline)
        adj, deg, e = _local_search(adj, deg, deadline)
        if e > best_edges:
            best_edges = e
            best_adj = [a for a in adj]
            best_deg = [d for d in deg]
        if best_edges >= 50:  # near complete, stop early
            break

    edges = _edges_from_adj(best_adj)
    return edges


if __name__ == "__main__":
    print(construct())
