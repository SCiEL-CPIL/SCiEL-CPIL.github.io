import numpy as np
import random
import time
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 147.0

    def remaining():
        return DEADLINE - (time.time() - start)

    # ---------- Helpers ----------
    def build_adj(n, bits):
        """bits: list of 0/1 per pair (i,j) i<j; 0 means i->j, 1 means j->i"""
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                b = bits[idx]; idx += 1
                if b == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        return adj

    def adj_to_bits(n, adj):
        bits = []
        for i in range(n):
            for j in range(i+1, n):
                bits.append(0 if adj[i][j] == 1 else 1)
        return bits

    def out_masks(n, adj):
        om = [0]*n
        for i in range(n):
            m = 0
            row = adj[i]
            for j in range(n):
                if i != j and row[j]:
                    m |= (1 << j)
            om[i] = m
        return om

    def count_violations(n, adj, om):
        cnt = 0
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
            ok = False
            for v in range(n):
                if v in (a,b,c,d):
                    continue
                if (om[v] & qm) == qm:
                    ok = True
                    break
            if not ok:
                cnt += 1
        return cnt

    def get_violating_quads(n, adj, om):
        V = []
        for quad in combinations(range(n), 4):
            a,b,c,d = quad
            qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
            ok = False
            for v in range(n):
                if v in (a,b,c,d):
                    continue
                if (om[v] & qm) == qm:
                    ok = True
                    break
            if not ok:
                V.append((a,b,c,d,qm))
        return V

    def paley_bits(p):
        sq = set((x*x) % p for x in range(1, p))
        bits = []
        for i in range(p):
            for j in range(i+1, p):
                if (j - i) % p in sq:
                    bits.append(0)   # i -> j
                else:
                    bits.append(1)   # j -> i
        return bits

    # ---------- Pruning ----------
    def prune(n, adj, om, max_rounds=25):
        """Iteratively remove vertices not uniquely dominating any 4-set."""
        active = list(range(n))
        for _ in range(max_rounds):
            if remaining() < 1.0:
                break
            # count unique dominations per active vertex
            unique = {v: 0 for v in active}
            active_set = set(active)
            for quad in combinations(active, 4):
                a,b,c,d = quad
                qm = (1<<a)|(1<<b)|(1<<c)|(1<<d)
                doms = []
                for v in active_set:
                    if v in (a,b,c,d):
                        continue
                    if (om[v] & qm) == qm:
                        doms.append(v)
                if len(doms) == 1:
                    unique[doms[0]] += 1
            removable = [v for v in active if unique[v] == 0]
            if not removable:
                break
            remove_set = set(removable)
            active = [v for v in active if v not in remove_set]
        return active

    # ---------- Error repair (local search) ----------
    def repair(n, adj, om, viol, steps_budget):
        """Flip edges to try to fix violations. Returns (adj, om, viol_count)."""
        n_quads = len(viol)
        for step in range(steps_budget):
            if remaining() < 1.0:
                break
            if n_quads == 0:
                break
            # pick a violating quad (weighted toward most under-dominated)
            quad = random.choice(viol)
            a,b,c,d,qm = quad
            # candidate dominators: vertices outside quad
            best_v = -1
            best_score = -1
            for v in range(n):
                if v in (a,b,c,d):
                    continue
                sc = (om[v] & qm).bit_count()
                if sc > best_score:
                    best_score = sc
                    best_v = v
            if best_v < 0 or best_score == 4:
                continue
            # flip edges from best_v to the quad's vertices
            changed = False
            for x in (a,b,c,d):
                if adj[best_v][x] == 0:
                    adj[best_v][x] = 1
                    adj[x][best_v] = 0
                    om[best_v] |= (1 << x)
                    om[x] &= ~(1 << best_v)
                    changed = True
            if not changed:
                continue
            # recompute violations (expensive but accurate)
            viol = get_violating_quads(n, adj, om)
            n_quads = len(viol)
        return n_quads

    # ---------- Main ----------
    best_result = None  # (n, bits)

    # Phase 1: Paley construction + prune + repair (primary strategy)
    for p in (83, 97, 89):
        if remaining() < 12.0:
            break
        bits = paley_bits(p)
        n = p
        adj = build_adj(n, bits)
        om = out_masks(n, adj)
        active = prune(n, adj, om, max_rounds=30)
        if len(active) < n:
            # relabel
            relab = {v: i for i, v in enumerate(active)}
            m = len(active)
            new_adj = [[0]*m for _ in range(m)]
            for i, u in enumerate(active):
                for j, w in enumerate(active):
                    if i != j:
                        new_adj[i][j] = adj[u][w]
            new_om = out_masks(m, new_adj)
            # repair on the pruned graph
            viol = get_violating_quads(m, new_adj, new_om)
            cnt = len(viol)
            if cnt == 0:
                candidate_bits = adj_to_bits(m, new_adj)
                if best_result is None or m < best_result[0]:
                    best_result = (m, candidate_bits)
                break
            elif cnt <= 2000:
                steps = max(200, int(remaining() * 30))
                cnt = repair(m, new_adj, new_om, viol, steps)
                if cnt == 0:
                    candidate_bits = adj_to_bits(m, new_adj)
                    if best_result is None or m < best_result[0]:
                        best_result = (m, candidate_bits)
                    break
        if remaining() < 2.0:
            break

    # Phase 2: If we have a good candidate, try to shrink further via targeted pruning
    if best_result is not None:
        n_best, bits_best = best_result
        # Already have a valid one; try to prune it further
        adj_b = build_adj(n_best, bits_best)
        om_b = out_masks(n_best, adj_b)
        active_b = prune(n_best, adj_b, om_b, max_rounds=15)
        if len(active_b) < n_best:
            relab = {v: i for i, v in enumerate(active_b)}
            m = len(active_b)
            new_adj = [[0]*m for _ in range(m)]
            for i, u in enumerate(active_b):
                for j, w in enumerate(active_b):
                    if i != j:
                        new_adj[i][j] = adj_b[u][w]
            new_om = out_masks(m, new_adj)
            viol = get_violating_quads(m, new_adj, new_om)
            if len(viol) == 0:
                candidate_bits = adj_to_bits(m, new_adj)
                if best_result is None or m < best_result[0]:
                    best_result = (m, candidate_bits)

    # Phase 3: Fallback — try small random tournaments (n from 31 to 40)
    if best_result is None:
        for n_try in range(31, 41):
            if remaining() < 3.0:
                break
            for attempt in range(5):
                if remaining() < 2.0:
                    break
                random.seed()
                bits = [random.choice([0,1]) for _ in range(n_try*(n_try-1)//2)]
                adj = build_adj(n_try, bits)
                om = out_masks(n_try, adj)
                cnt = count_violations(n_try, adj, om)
                if cnt == 0:
                    best_result = (n_try, bits)
                    break
                if cnt < 200:
                    viol = get_violating_quads(n_try, adj, om)
                    steps = max(100, int(remaining() * 20))
                    cnt = repair(n_try, adj, om, viol, steps)
                    if cnt == 0:
                        best_result = (n_try, bits)
                        break

    # Phase 4: Absolute fallback — Paley 83 (guaranteed valid for k=4)
    if best_result is None:
        p = 83
        bits = paley_bits(p)
        # quick verify
        adj = build_adj(p, bits)
        om = out_masks(p, adj)
        cnt = count_violations(p, adj, om)
        if cnt == 0:
            best_result = (p, bits)
        else:
            # Should not happen; try a different prime
            for p2 in (89, 97):
                bits2 = paley_bits(p2)
                adj2 = build_adj(p2, bits2)
                om2 = out_masks(p2, adj2)
                if count_violations(p2, adj2, om2) == 0:
                    best_result = (p2, bits2)
                    break

    if best_result is None:
        # last resort
        p = 83
        bits = paley_bits(p)
        best_result = (p, bits)

    n_final, bits_final = best_result
    return {'n': n_final, 'orientation': bits_final}
