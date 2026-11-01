import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 110.0
    
    total_words = q ** n
    
    # Precompute spheres and coverers
    spheres = [[] for _ in range(total_words)]
    coverers = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        word = [(i // (q ** (n - 1 - j))) % q for j in range(n)]
        covered = [i]
        for pos in range(n):
            for delta in [-1, 1]:
                new_word = word.copy()
                new_word[pos] = (word[pos] + delta) % q
                new_idx = sum(new_word[j] * (q ** (n - 1 - j)) for j in range(n))
                covered.append(new_idx)
        spheres[i] = covered
        for w in covered:
            coverers[w].append(i)
            
    # Convert coverers to sets for fast intersection
    coverers_sets = [set(c) for c in coverers]

    # Initial code: Direct product (243 words)
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a + b) % 3, (a + 2 * b) % 3))
            
    initial_code = set()
    for h in tetracode:
        for tail in range(3 ** (n - 4)):
            t0 = (tail // 9) % 3
            t1 = (tail // 3) % 3
            t2 = tail % 3
            word = h + (t0, t1, t2)
            idx = sum(word[j] * (q ** (n - 1 - j)) for j in range(n))
            initial_code.add(idx)
            
    best_code = list(initial_code)
    best_size = len(best_code)
    
    def get_coverage(code_set):
        cov = np.zeros(total_words, dtype=np.int32)
        for c in code_set:
            for w in spheres[c]:
                cov[w] += 1
        return cov
        
    def count_slack(cov):
        # Number of words with coverage > 1
        return np.sum(cov > 1)

    current_code = set(initial_code)
    current_cov = get_coverage(current_code)
    current_slack = count_slack(current_cov)
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() - start_time < time_limit:
        # 1. Try Removals
        removed_any = False
        # Find all removable
        removable = []
        for c in current_code:
            can_remove = True
            for w in spheres[c]:
                if current_cov[w] == 1:
                    can_remove = False
                    break
            if can_remove:
                removable.append(c)
                
        if removable:
            c_to_remove = random.choice(removable)
            current_code.remove(c_to_remove)
            for w in spheres[c_to_remove]:
                current_cov[w] -= 1
            removed_any = True
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue # Restart loop to re-evaluate removals
        
        # 2. Try Swaps
        # Pick a random codeword to remove
        if current_code:
            c_old = random.choice(list(current_code))
            current_code.remove(c_old)
            for w in spheres[c_old]:
                current_cov[w] -= 1
                
            uncovered = np.where(current_cov == 0)[0]
            
            if len(uncovered) > 0:
                # Find candidates
                candidates = coverers_sets[uncovered[0]].copy()
                for w in uncovered[1:]:
                    candidates &= coverers_sets[w]
                candidates -= current_code
                
                if candidates:
                    # Pick best candidate based on slack increase
                    best_cand = None
                    best_slack = -1
                    # Sample if too many candidates
                    cand_list = list(candidates)
                    if len(cand_list) > 50:
                        cand_list = random.sample(cand_list, 50)
                        
                    for c_new in cand_list:
                        # Calculate new slack
                        # We need to know how many words in spheres[c_new] will have cov > 1
                        # Current cov for w in spheres[c_new] is current_cov[w]
                        # New cov will be current_cov[w] + 1
                        # Slack increases for w where current_cov[w] == 1
                        # Slack decreases for w where current_cov[w] == 0 (becomes 1)
                        # But we know current_cov[w] > 0 for all w not in uncovered?
                        # No, uncovered are 0.
                        # For w in spheres[c_new]:
                        #   if w in uncovered: new_cov=1, old=0. Slack change: 0 (1 is not >1)
                        #   if w not in uncovered: new_cov=old+1.
                        #     if old==1: new=2. Slack +1.
                        #     if old>1: new>2. Slack 0 (was already >1).
                        # So slack change = count(w in spheres[c_new] such that current_cov[w] == 1)
                        
                        slack_change = 0
                        for w in spheres[c_new]:
                            if current_cov[w] == 1:
                                slack_change += 1
                            
                        # We want to maximize slack
                        if slack_change > best_slack:
                            best_slack = slack_change
                            best_cand = c_new
                            
                    if best_cand is not None:
                        # Apply swap
                        current_code.add(best_cand)
                        for w in spheres[best_cand]:
                            current_cov[w] += 1
                        new_slack = count_slack(current_cov)
                        
                        if new_slack > current_slack:
                            current_slack = new_slack
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        removed_any = True # Mark as activity
                    else:
                        # Rollback
                        current_code.add(c_old)
                        for w in spheres[c_old]:
                            current_cov[w] += 1
                        no_improve_count += 1
                else:
                    # Rollback
                    current_code.add(c_old)
                    for w in spheres[c_old]:
                        current_cov[w] += 1
                    no_improve_count += 1
            else:
                # No uncovered words? That means c_old was removable.
                # But we checked removable above.
                # This can happen if random choice in removable missed it? No.
                # So this case should be rare.
                current_code.add(c_old)
                for w in spheres[c_old]:
                    current_cov[w] += 1
                no_improve_count += 1
        else:
            break
            
        if not removed_any:
            # Stuck
            no_improve_count += 10
            if no_improve_count > max_no_improve:
                # Restart from best
                current_code = set(best_code)
                current_cov = get_coverage(current_code)
                current_slack = count_slack(current_cov)
                no_improve_count = 0
                # Perturb
                for _ in range(5):
                    if len(current_code) <= 10: break
                    c_to_remove = random.choice(list(current_code))
                    current_code.remove(c_to_remove)
                    for w in spheres[c_to_remove]:
                        current_cov[w] -= 1
                    uncovered = np.where(current_cov == 0)[0]
                    if len(uncovered) > 0:
                        candidates = coverers_sets[uncovered[0]].copy()
                        for w in uncovered[1:]:
                            candidates &= coverers_sets[w]
                        candidates -= current_code
                        if candidates:
                            c_new = random.choice(list(candidates))
                            current_code.add(c_new)
                            for w in spheres[c_new]:
                                current_cov[w] += 1
                        else:
                            current_code.add(c_to_remove)
                            for w in spheres[c_to_remove]:
                                current_cov[w] += 1
                            break

    return [[(c // (q ** (n - 1 - j))) % q for j in range(n)] for c in best_code]
