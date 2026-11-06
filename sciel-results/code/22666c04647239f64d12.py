"""Stronger snake search for Q_9: fast forbidden-set check + randomized multi-restart
with limited in-place backtracking. Returns the longest induced path (snake) found."""
import time
import random

N = 9
DEADLINE = 130.0  # safety margin under 150s kill

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    best_path = None
    best_len = 0
    rng = random.Random(12345)

    # Precompute neighbor lists for speed
    NEIGH = [[v ^ (1 << b) for b in range(N)] for v in range(1 << N)]

    def elapsed():
        return time.time() - start

    def greedy_extend(path, used, forbidden, rng):
        """Extend greedily; mutates in place. Returns new length."""
        while True:
            if elapsed() > DEADLINE - 5.0:
                break
            cur = path[-1]
            cands = NEIGH[cur]
            valid = [v for v in cands if v not in used and v not in forbidden]
            if not valid:
                break
            if rng.random() < 0.7:
                nxt = valid[0]
            else:
                nxt = rng.choice(valid)
            path.append(nxt)
            used.add(nxt)
            for nb in NEIGH[nxt]:
                forbidden.add(nb)
        return len(path)

    def backtrack_improve(path, used, forbidden, rng, depth_budget=25):
        """In-place DFS with undo, limited depth. Returns best path copy found."""
        nonlocal best_path, best_len
        best_local = list(path)
        best_local_len = len(path)

        def dfs(pth, used_s, forb_s, budget):
            nonlocal best_local, best_local_len
            if elapsed() > DEADLINE - 5.0:
                return
            cur = pth[-1]
            cands = NEIGH[cur]
            valid = [v for v in cands if v not in used_s and v not in forb_s]
            if len(pth) > best_local_len:
                best_local_len = len(pth)
                best_local = list(pth)
            if not valid or budget <= 0:
                return
            rng.shuffle(valid)
            for v in valid:
                if elapsed() > DEADLINE - 5.0:
                    return
                # add v
                pth.append(v)
                used_s.add(v)
                added_forb = []
                for nb in NEIGH[v]:
                    if nb not in forb_s:
                        forb_s.add(nb)
                        added_forb.append(nb)
                dfs(pth, used_s, forb_s, budget - 1)
                # undo
                pth.pop()
                used_s.discard(v)
                for nb in added_forb:
                    forb_s.discard(nb)
                if budget <= 1:
                    break

        dfs(path, used, forbidden, depth_budget)
        if best_local_len > best_len:
            best_len = best_local_len
            best_path = list(best_local)

    max_restarts = 200000
    for r in range(max_restarts):
        if elapsed() > DEADLINE - 5.0:
            break
        start_v = rng.randrange(1 << N)
        path = [start_v]
        used = {start_v}
        forbidden = set(NEIGH[start_v])
        cur_len = greedy_extend(path, used, forbidden, rng)
        if cur_len > best_len:
            best_len = cur_len
            best_path = list(path)
        if rng.random() < 0.25 and elapsed() < DEADLINE - 15.0:
            backtrack_improve(path, used, forbidden, rng, depth_budget=25)

    if best_path is None:
        best_path = [0]
    return best_path

# EVOLVE-BLOCK-END
