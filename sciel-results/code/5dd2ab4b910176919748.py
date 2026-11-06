import random
import time
from itertools import combinations

def construct():
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Precompute K(2,2,2) configurations
    # Each config is defined by 3 disjoint pairs, yielding 8 transversal triples
    triple_to_kconfigs = {t: [] for t in all_triples}
    kconfigs = []
    
    def partitions_into_pairs(points):
        a = points[0]
        for i in range(1, 6):
            b = points[i]
            remaining = [p for j, p in enumerate(points) if j not in (0, i)]
            c, d, e, f = remaining
            yield (a, b), (c, d), (e, f)
            yield (a, b), (c, e), (d, f)
            yield (a, b), (c, f), (d, e)
    
    for six_points in combinations(range(n), 6):
        for pairs in partitions_into_pairs(six_points):
            (a, a_prime), (b, b_prime), (c, c_prime) = pairs
            transversals_canon = set()
            for x in (a, a_prime):
                for y in (b, b_prime):
                    for z in (c, c_prime):
                        transversals_canon.add(tuple(sorted((x, y, z))))
            kconfigs.append(transversals_canon)
            for t in transversals_canon:
                triple_to_kconfigs[t].append(len(kconfigs) - 1)
    
    num_kconfigs = len(kconfigs)
    
    # For each kconfig, store the list of 8 triples
    kconfig_triples = [list(k) for k in kconfigs]
    
    def greedy_solve(order):
        present = set()
        kconfig_counts = [0] * num_kconfigs
        blocked = set()
        
        for t in order:
            if t in blocked:
                continue
            # Check if still valid (should be, but just in case)
            # Actually, if t is not in blocked, it means no kconfig containing t is at count 7.
            # So we can add it.
            present.add(t)
            for kidx in triple_to_kconfigs[t]:
                kconfig_counts[kidx] += 1
                if kconfig_counts[kidx] == 7:
                    # Find the missing triple in this kconfig
                    # The missing one is the one in kconfig_triples[kidx] not in present
                    # Since we just added t, and count is 7, exactly one is missing.
                    for u in kconfig_triples[kidx]:
                        if u not in present:
                            blocked.add(u)
                            break
        return present, kconfig_counts, blocked

    def local_improve(present, kconfig_counts, blocked):
        improved = True
        while improved:
            improved = False
            # Find a candidate not in present and not in blocked
            # We can iterate through all_triples or maintain a list of candidates.
            # For simplicity and speed, let's just pick randomly from available or iterate.
            # Iterating through all 560 is fast enough if we do it a few times.
            # But we can be smarter: keep a list of candidates.
            
            # Let's just iterate through all triples to find one that is not present and not blocked.
            # This is O(N).
            for t in all_triples:
                if t in present or t in blocked:
                    continue
                # Add t
                present.add(t)
                for kidx in triple_to_kconfigs[t]:
                    kconfig_counts[kidx] += 1
                    if kconfig_counts[kidx] == 7:
                        for u in kconfig_triples[kidx]:
                            if u not in present:
                                blocked.add(u)
                                break
                improved = True
                break # Restart loop to check for new unblocks? No, adding only blocks more.
                # Actually, adding a triple never unblocks anything. It only blocks.
                # So we can just continue scanning?
                # If we add t, we might block u. But u was already not in present.
                # We can just continue the loop from where we left off?
                # Or just break and restart to be safe?
                # Restarting is safer but slower.
                # Let's just continue.
                # Wait, if I add t, I block u. u might be later in the list.
                # So I should check if u is in blocked before trying to add it.
                # Since I update blocked immediately, the `if t in blocked` check will catch it.
                # So I can just continue the loop.
        
        return present, kconfig_counts, blocked

    # Initial best solution
    best_count = 0
    best_triples = None
    
    # Warm start with greedy
    # Sort by degree (number of kconfigs) to prioritize less constrained triples?
    # Or random?
    random.seed(42)
    # Let's try sorting by degree ascending
    degrees = {t: len(triple_to_kconfigs[t]) for t in all_triples}
    ordered_triples = sorted(all_triples, key=lambda x: degrees[x])
    
    present, kconfig_counts, blocked = greedy_solve(ordered_triples)
    present, kconfig_counts, blocked = local_improve(present, kconfig_counts, blocked)
    best_count = len(present)
    best_triples = [list(t) for t in present]
    
    # ILS Search
    start_time = time.time()
    time_limit = 110.0
    
    current_present = set(present)
    current_kconfig_counts = kconfig_counts[:]
    current_blocked = set(blocked)
    
    while time.time() - start_time < time_limit:
        # Perturb: remove k random triples from present
        k = random.randint(2, 5)
        to_remove = random.sample(list(current_present), min(k, len(current_present)))
        
        # Create a copy to perturb
        perturbed_present = set(current_present)
        perturbed_kconfig_counts = current_kconfig_counts[:]
        perturbed_blocked = set(current_blocked)
        
        for t in to_remove:
            perturbed_present.remove(t)
            for kidx in triple_to_kconfigs[t]:
                perturbed_kconfig_counts[kidx] -= 1
                if perturbed_kconfig_counts[kidx] == 6:
                    # Find the missing triple in this kconfig
                    # It was blocked, now it's unblocked
                    for u in kconfig_triples[kidx]:
                        if u not in perturbed_present:
                            if u in perturbed_blocked:
                                perturbed_blocked.remove(u)
                            break
        
        # Recover: local improve
        perturbed_present, perturbed_kconfig_counts, perturbed_blocked = local_improve(perturbed_present, perturbed_kconfig_counts, perturbed_blocked)
        
        # Check if better
        if len(perturbed_present) > best_count:
            best_count = len(perturbed_present)
            best_triples = [list(t) for t in perturbed_present]
            
        # Update current state
        current_present = perturbed_present
        current_kconfig_counts = perturbed_kconfig_counts
        current_blocked = perturbed_blocked
        
    return best_triples
