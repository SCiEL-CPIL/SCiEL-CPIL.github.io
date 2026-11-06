"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin from 90s limit

    MAX_N = 512
    MAX_SIZE = 48
    MAX_SUM = 24576  # 48 * 512

    # --- Fast validity check for adding x to selected set ---
    def is_valid_addition(x, selected, dp_all, dp_without, n):
        """
        Check if x can be added to selected without violating the mean condition.
        Returns True if valid.
        """
        # Check 1: Is x the mean of any subset of selected of size >= 2?
        for k in range(2, n + 1):
            target = k * x
            if target > MAX_SUM:
                break
            if k < len(dp_all) and ((dp_all[k] >> target) & 1):
                return False

        # Check 2: For each a in selected, is a the mean of a subset involving x?
        for a in selected:
            dp_w = dp_without[a]
            for k in range(2, n + 1):
                target = k * a - x
                if target < 0:
                    continue
                if target > (k - 1) * 512:
                    continue
                if k - 1 < len(dp_w) and ((dp_w[k - 1] >> target) & 1):
                    return False

        return True

    # --- Fast Greedy Construction with fixed order ---
    def try_greedy(order):
        """
        Greedily builds a valid set from a fixed order.
        Returns the set of selected integers.
        """
        selected = []
        n = 0
        dp_all = [1]
        dp_without = {}

        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= MAX_SIZE:
                break

            if not is_valid_addition(x, selected, dp_all, dp_without, n):
                continue

            # Add x to selected
            selected.append(x)

            # Update dp_all
            old_dp_all = dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n:
                    val |= old_dp_all[k]
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k - 1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all

            # Update dp_without
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 1)
                new_list[0] = 1
                for k in range(1, n + 1):
                    val = 0
                    if k <= n - 1:
                        val |= old_dp_w[k]
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list

            # For new element x, dp_without[x] is the old dp_all (set before adding x)
            new_dp_without[x] = list(old_dp_all)

            dp_without = new_dp_without
            n += 1

        return selected

    # --- Dynamic Conflict-Density Greedy ---
    def try_dynamic_greedy(initial_candidates, top_k_score=50):
        """
        At each step, score all remaining candidates by conflict density
        (number of k*x targets that are achievable in current dp_all),
        pick the lowest-conflict valid candidate.
        
        To keep runtime manageable, we only fully score a subset of candidates:
        those with the lowest cheap proxy scores.
        """
        selected = []
        n = 0
        dp_all = [1]
        dp_without = {}
        remaining = list(initial_candidates)
        remaining_set = set(remaining)

        while n < MAX_SIZE and remaining:
            if time.time() - start_time > deadline:
                break

            # Score candidates by conflict density
            # Cheap proxy: count how many k*x are in dp_all[k] for k=2..n
            # For speed, only check k=2,3,4 (most common conflicts)
            scored = []
            for x in remaining:
                conflict_count = 0
                # Check k=2: is 2*x achievable with 2 elements?
                if n >= 2:
                    target = 2 * x
                    if target <= MAX_SUM and ((dp_all[2] >> target) & 1):
                        conflict_count += 10  # Weight k=2 more heavily
                # Check k=3: is 3*x achievable with 3 elements?
                if n >= 3:
                    target = 3 * x
                    if target <= MAX_SUM and ((dp_all[3] >> target) & 1):
                        conflict_count += 5
                # Check k=4: is 4*x achievable with 4 elements?
                if n >= 4:
                    target = 4 * x
                    if target <= MAX_SUM and ((dp_all[4] >> target) & 1):
                        conflict_count += 3
                # Check if x itself causes issues with existing elements (quick check)
                # For each a in selected, check if a is mean of subset involving x
                # This is expensive, so skip for scoring and do full check later
                scored.append((conflict_count, x))

            # Sort by conflict count, take top_k_score candidates
            scored.sort()
            candidates_to_check = [x for _, x in scored[:top_k_score]]

            # Find the first valid candidate
            chosen = None
            for x in candidates_to_check:
                if is_valid_addition(x, selected, dp_all, dp_without, n):
                    chosen = x
                    break

            if chosen is None:
                # No candidate in top_k is valid; try all remaining (expensive fallback)
                for x in remaining:
                    if is_valid_addition(x, selected, dp_all, dp_without, n):
                        chosen = x
                        break

            if chosen is None:
                break

            # Remove chosen from remaining
            remaining.remove(chosen)
            remaining_set.discard(chosen)

            # Add chosen to selected
            selected.append(chosen)

            # Update dp_all
            old_dp_all = dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = 0
                if k <= n:
                    val |= old_dp_all[k]
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k - 1] << chosen)
                new_dp_all[k] = val
            dp_all = new_dp_all

            # Update dp_without
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 1)
                new_list[0] = 1
                for k in range(1, n + 1):
                    val = 0
                    if k <= n - 1:
                        val |= old_dp_w[k]
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << chosen)
                    new_list[k] = val
                new_dp_without[a] = new_list

            # For new element, dp_without[new] is the old dp_all
            new_dp_without[chosen] = list(old_dp_all)

            dp_without = new_dp_without
            n += 1

        return selected

    # --- Conflict Graph Model ---
    def build_conflict_graph():
        """
        Compute risk scores based on 3-AP participation.
        """
        risk = [0] * (MAX_N + 1)
        for w in range(1, MAX_N + 1):
            for u in range(1, MAX_N + 1):
                v = 2 * w - u
                if 1 <= v <= MAX_N and v != u and v != w:
                    risk[u] += 1
                    risk[v] += 1
                    risk[w] += 1
        return risk

    # --- Main Search Logic ---
    best_set = []
    best_len = 0

    # 1. Compute risk scores
    risk_scores = build_conflict_graph()

    # Strategy 1: Dynamic conflict-density greedy (primary)
    # Start with all candidates; dynamic scoring will pick the best at each step
    res = try_dynamic_greedy(list(range(1, MAX_N + 1)), top_k_score=50)
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Dynamic greedy with low-risk bias in initial ordering
    if time.time() - start_time < deadline - 5:
        candidates_by_risk = sorted(range(1, MAX_N + 1), key=lambda x: risk_scores[x])
        res = try_dynamic_greedy(candidates_by_risk, top_k_score=40)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Dynamic greedy with high-risk bias
    if time.time() - start_time < deadline - 5:
        candidates_by_risk_desc = sorted(range(1, MAX_N + 1), key=lambda x: -risk_scores[x])
        res = try_dynamic_greedy(candidates_by_risk_desc, top_k_score=40)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 4: Fixed-order greedy (fallback, fast)
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(1, MAX_N + 1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 5: Randomized dynamic restarts
    for i in range(30):
        if time.time() - start_time > deadline - 2:
            break
        if best_len >= MAX_SIZE:
            break

        random.seed(i * 98765 + 43210)
        # Blend: sort by risk + noise
        candidates = list(range(1, MAX_N + 1))
        candidates.sort(key=lambda x: risk_scores[x] + random.random() * 200)
        res = try_dynamic_greedy(candidates, top_k_score=35)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= MAX_SIZE:
                break

    # Strategy 6: Modular orders with fixed greedy
    for mod in [7, 11, 13, 17]:
        if time.time() - start_time > deadline - 3:
            break
        buckets = [[] for _ in range(mod)]
        for x in range(1, MAX_N + 1):
            buckets[x % mod].append(x)
        order = []
        max_len = max(len(b) for b in buckets) if buckets else 0
        for i in range(max_len):
            for b in buckets:
                if i < len(b):
                    order.append(b[i])
        res = try_greedy(order)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= MAX_SIZE:
                break

    # Strategy 7: Pure random fixed-order restarts
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 3:
        for i in range(30):
            if time.time() - start_time > deadline - 1:
                break
            if best_len >= MAX_SIZE:
                break

            random.seed(123456 + i * 7919)
            candidates = list(range(1, MAX_N + 1))
            random.shuffle(candidates)
            res = try_greedy(candidates)
            if len(res) > best_len:
                best_set = res
                best_len = len(res)

    return best_set

# EVOLVE-BLOCK-END
