import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s limit

    N = 256
    MAX_SIZE = 16  # Max elements per set

    best_size = 0
    best_A = []
    best_B = []

    def try_add(x, S, sums_mask, diffs_other_mask):
        """Try to add x to set S. Returns (success, new_sums, new_diffs)."""
        if x in S:
            return (False, 0, 0)
        # Check sum conflicts
        temp_sums = 0
        for a in S:
            s = x + a
            if (sums_mask >> s) & 1:
                return (False, 0, 0)
            temp_sums |= (1 << s)
        if (sums_mask >> (2 * x)) & 1:
            return (False, 0, 0)
        temp_sums |= (1 << (2 * x))
        # Check diff conflicts with other set's diffs
        temp_diffs = 0
        for a in S:
            d = abs(x - a)
            if d > 0:
                if (diffs_other_mask >> d) & 1:
                    return (False, 0, 0)
                temp_diffs |= (1 << d)
        return (True, temp_sums, temp_diffs)

    def compute_masks(S):
        """Compute sums and diffs masks for set S."""
        sums = 0
        diffs = 0
        for i, a in enumerate(S):
            sums |= (1 << (2 * a))
            for j in range(i):
                s = a + S[j]
                sums |= (1 << s)
                d = abs(a - S[j])
                if d > 0:
                    diffs |= (1 << d)
        return sums, diffs

    def beam_search_pass(beam_width=100):
        """Beam search: keep top beam_width states by total size."""
        # Initial state: empty A, empty B
        # State: (A, B, sums_A, sums_B, diffs_A, diffs_B)
        states = [(list(), list(), 0, 0, 0, 0)]
        order = list(range(1, N + 1))
        random.shuffle(order)

        for x in order:
            new_states = []
            for (A, B, sums_A, sums_B, diffs_A, diffs_B) in states:
                # Try adding x to A
                if len(A) < MAX_SIZE:
                    success, ns, nd = try_add(x, A, sums_A, diffs_B)
                    if success:
                        new_A = A + [x]
                        new_states.append((new_A, B, sums_A | ns, sums_B, diffs_A | nd, diffs_B))

                # Try adding x to B
                if len(B) < MAX_SIZE:
                    success, ns, nd = try_add(x, B, sums_B, diffs_A)
                    if success:
                        new_B = B + [x]
                        new_states.append((A, new_B, sums_A, sums_B | ns, diffs_A, diffs_B | nd))

                # Option: skip x (keep current state)
                new_states.append((A, B, sums_A, sums_B, diffs_A, diffs_B))

            # Keep top beam_width by total size, with some diversity
            if len(new_states) > beam_width:
                # Sort by size descending, take top beam_width
                new_states.sort(key=lambda s: len(s[0]) + len(s[1]), reverse=True)
                # Keep top beam_width but ensure some diversity
                # Take top 70% by size, fill rest with random from remaining
                top_n = int(beam_width * 0.7)
                top_states = new_states[:top_n]
                remaining = new_states[top_n:]
                if remaining:
                    fill_n = beam_width - len(top_states)
                    random.shuffle(remaining)
                    top_states.extend(remaining[:fill_n])
                new_states = top_states

            states = new_states

            if time.time() > deadline - 5.0:
                break

        # Return best state
        if not states:
            return [], []
        states.sort(key=lambda s: len(s[0]) + len(s[1]), reverse=True)
        return states[0][0], states[0][1]

    def polish(A, B, time_limit):
        """Local search polish: try insertions and swaps."""
        A = list(A)
        B = list(B)
        polish_deadline = time.time() + time_limit
        improved = True

        while improved and time.time() < polish_deadline:
            improved = False
            sums_A, diffs_A = compute_masks(A)
            sums_B, diffs_B = compute_masks(B)
            all_elements = set(A) | set(B)
            available = [x for x in range(1, N + 1) if x not in all_elements]

            # Try insertions into A
            for x in available:
                if len(A) >= MAX_SIZE:
                    break
                success, ns, nd = try_add(x, A, sums_A, diffs_B)
                if success:
                    A.append(x)
                    sums_A |= ns
                    diffs_A |= nd
                    improved = True
                    break

            if improved:
                continue

            # Try insertions into B
            sums_A, diffs_A = compute_masks(A)
            sums_B, diffs_B = compute_masks(B)
            all_elements = set(A) | set(B)
            available = [x for x in range(1, N + 1) if x not in all_elements]
            for x in available:
                if len(B) >= MAX_SIZE:
                    break
                success, ns, nd = try_add(x, B, sums_B, diffs_A)
                if success:
                    B.append(x)
                    sums_B |= ns
                    diffs_B |= nd
                    improved = True
                    break

            if improved:
                continue

            # Try swaps: remove one element from A, try to add a different one
            for i in range(len(A)):
                saved = A[i]
                A.pop(i)
                sums_A, diffs_A = compute_masks(A)
                all_elements = set(A) | set(B)
                available = [x for x in range(1, N + 1) if x not in all_elements]
                best_new = None
                for x in available:
                    if x == saved:
                        continue
                    success, ns, nd = try_add(x, A, sums_A, diffs_B)
                    if success:
                        best_new = x
                        break
                if best_new is not None:
                    A.append(best_new)
                    improved = True
                    break
                else:
                    A.insert(i, saved)  # Restore

            if improved:
                continue

            # Try swaps: remove one element from B, try to add a different one
            for i in range(len(B)):
                saved = B[i]
                B.pop(i)
                sums_B, diffs_B = compute_masks(B)
                all_elements = set(A) | set(B)
                available = [x for x in range(1, N + 1) if x not in all_elements]
                best_new = None
                for x in available:
                    if x == saved:
                        continue
                    success, ns, nd = try_add(x, B, sums_B, diffs_A)
                    if success:
                        best_new = x
                        break
                if best_new is not None:
                    B.append(best_new)
                    improved = True
                    break
                else:
                    B.insert(i, saved)  # Restore

        return A, B

    # Run multiple beam search passes with different random seeds
    while time.time() < deadline - 10.0:
        A, B = beam_search_pass(beam_width=80)
        current_size = len(A) + len(B)

        # Polish the result
        if current_size > best_size or (current_size >= best_size - 2 and random.random() < 0.4):
            polish_time = min(3.0, max(0.5, (deadline - 5.0) - time.time()))
            if polish_time > 0.5:
                A_pol, B_pol = polish(A, B, polish_time)
                current_size = len(A_pol) + len(B_pol)
                if current_size > best_size:
                    best_size = current_size
                    best_A = list(A_pol)
                    best_B = list(B_pol)
            else:
                if current_size > best_size:
                    best_size = current_size
                    best_A = list(A)
                    best_B = list(B)

        if best_size >= 32:
            break

    # Final polish on best solution with remaining time
    remaining = deadline - 3.0 - time.time()
    if remaining > 1.0 and best_A:
        A_pol, B_pol = polish(best_A, best_B, remaining)
        if len(A_pol) + len(B_pol) > best_size:
            best_A = A_pol
            best_B = B_pol
            best_size = len(A_pol) + len(B_pol)

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
