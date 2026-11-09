import random
import time

def construct():
    def score(S):
        present = [False] * 513
        for x in S:
            present[x] = True
            
        diffs = set()
        list_S = list(S)
        n = len(list_S)
        
        for i in range(n):
            a = list_S[i]
            for j in range(i+1, n):
                c = list_S[j]
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if present[b]:
                        d = c - a
                        diffs.add(d)
        return len(diffs)

    def greedy_construct():
        S = set()
        candidates = list(range(1, 513))
        
        for _ in range(32):
            best_x = None
            best_score = -1
            # Shuffle to break ties randomly
            random.shuffle(candidates)
            for x in candidates:
                if x in S:
                    continue
                new_S = S | {x}
                s = score(new_S)
                if s > best_score:
                    best_score = s
                    best_x = x
            S.add(best_x)
            
        return S

    def local_search(S, time_limit):
        current_S = set(S)
        current_score = score(current_S)
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick a random element to replace
            u = random.choice(list(current_S))
            # Try all possible replacements
            best_v = None
            best_new_score = current_score
            candidates = [x for x in range(1, 513) if x not in current_S]
            random.shuffle(candidates)
            
            for v in candidates:
                new_S = (current_S - {u}) | {v}
                new_score = score(new_S)
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_v = v
            
            if best_v is not None:
                current_S.remove(u)
                current_S.add(best_v)
                current_score = best_new_score
            else:
                # If no improvement found in this pass, we might be stuck in a local optimum.
                # To escape, we can do a random move (Simulated Annealing style) or just continue.
                # Given the time limit, simple hill climbing might get stuck.
                # Let's add a small probability of accepting a worse move?
                # Or just restart the local search with a perturbation.
                # For simplicity and robustness, let's just continue trying different u's.
                # If we cycle, we waste time. Let's track visited states? Too complex.
                # Let's just break if we haven't improved in a long time?
                # Actually, the loop picks a random u each time, so it will explore.
                pass

        return current_S, current_score

    # 1. Greedy Construction
    S = greedy_construct()
    
    # 2. Local Search
    # We have about 100 seconds left.
    # Greedy takes ~1-2 seconds.
    # Local search: Each iteration is ~512 score calls.
    # Score call is ~500 ops.
    # 512 * 500 = 256,000 ops.
    # Python can do ~10M-50M simple ops/sec.
    # So ~5-10 iterations per second? Maybe slower due to overhead.
    # Let's say 100 iterations per second.
    # 100 seconds * 100 iter/sec = 10,000 iterations.
    # This should be enough to improve significantly.
    
    S, final_score = local_search(S, 110)
    
    return list(S)
