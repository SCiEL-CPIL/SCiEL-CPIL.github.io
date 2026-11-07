import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill

    def get_subset_sums(elements):
        """Get all non-empty subset sums as a set."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            sums.add(s)
        return sums

    def check_divisibility_conflict(sums):
        """Check if any two distinct sums have one dividing the other."""
        sorted_sums = sorted(sums)
        for i, a in enumerate(sorted_sums):
            for b in sorted_sums[i+1:]:
                if b % a == 0:
                    return True
        return False

    def is_valid(elements):
        """Check if the set is valid: unique subset sums and no divisibility conflicts."""
        n = len(elements)
        sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        return not check_divisibility_conflict(sums)

    def try_add(elements, val):
        """Try to add val to elements. Return (success, new_elements, new_sums) or (False, None, None)."""
        if val in elements:
            return False, None, None
        new_elements = elements + [val]
        n = len(new_elements)
        sums = set()
        # Compute all subset sums
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += new_elements[i]
            if s in sums:
                return False, None, None
            sums.add(s)
        if check_divisibility_conflict(sums):
            return False, None, None
        return True, new_elements, sums

    def try_remove(elements, idx, sums):
        """Try to remove element at idx. Return (new_elements, new_sums)."""
        new_elements = elements[:idx] + elements[idx+1:]
        n = len(new_elements)
        new_sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += new_elements[i]
            new_sums.add(s)
        return new_elements, new_sums

    # Initial greedy construction
    def greedy_construct():
        elements = []
        sums = set()
        for val in range(1, 257):
            if time.time() > deadline:
                break
            success, new_elements, new_sums = try_add(elements, val)
            if success:
                elements = new_elements
                sums = new_sums
        return elements, sums

    # Simulated annealing
    def simulated_annealing(initial_elements, initial_sums, deadline):
        if not initial_elements:
            return initial_elements

        current_elements = list(initial_elements)
        current_sums = set(initial_sums)
        best_elements = list(current_elements)
        best_size = len(current_elements)

        # Temperature schedule
        t_start = 1.0
        t_end = 0.01
        cooling = 0.995

        temperature = t_start
        no_improve_count = 0
        max_no_improve = 200

        while temperature > t_end and time.time() < deadline:
            if no_improve_count > max_no_improve:
                break

            # Choose a random move
            move_type = random.random()
            if move_type < 0.4:
                # Add a random element
                candidates = [v for v in range(1, 257) if v not in current_elements]
                if not candidates:
                    continue
                val = random.choice(candidates)
                success, new_elements, new_sums = try_add(current_elements, val)
                if not success:
                    no_improve_count += 1
                    continue
                # Accept if it improves size, or with some probability
                if len(new_elements) > len(current_elements):
                    current_elements = new_elements
                    current_sums = new_sums
                    no_improve_count = 0
                    if len(current_elements) > best_size:
                        best_elements = list(current_elements)
                        best_size = len(current_elements)
                else:
                    no_improve_count += 1
            elif move_type < 0.7:
                # Remove a random element
                if len(current_elements) <= 1:
                    continue
                idx = random.randint(0, len(current_elements) - 1)
                new_elements, new_sums = try_remove(current_elements, idx, current_sums)
                # Accept removal with some probability (to escape local optima)
                if random.random() < temperature * 0.5:
                    current_elements = new_elements
                    current_sums = new_sums
                    no_improve_count += 1
            else:
                # Swap: remove one, add another
                if len(current_elements) < 1:
                    continue
                idx = random.randint(0, len(current_elements) - 1)
                candidates = [v for v in range(1, 257) if v not in current_elements]
                if not candidates:
                    continue
                val = random.choice(candidates)
                # Remove idx first
                temp_elements = current_elements[:idx] + current_elements[idx+1:]
                temp_sums = set()
                n = len(temp_elements)
                for mask in range(1, 1 << n):
                    s = 0
                    for i in range(n):
                        if mask & (1 << i):
                            s += temp_elements[i]
                    temp_sums.add(s)
                # Try to add val
                success, new_elements, new_sums = try_add(temp_elements, val)
                if success:
                    if random.random() < temperature * 0.8:
                        current_elements = new_elements
                        current_sums = new_sums
                        if len(current_elements) > best_size:
                            best_elements = list(current_elements)
                            best_size = len(current_elements)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1

            temperature *= cooling

        return best_elements

    # First, do greedy construction
    initial_elements, initial_sums = greedy_construct()
    best_result = list(initial_elements)

    if time.time() > deadline:
        return [int(x) for x in best_result]

    # Run simulated annealing with multiple restarts
    random.seed(42)
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Perturb the current best
        if len(best_result) > 1:
            # Remove 1-3 random elements
            n_remove = min(random.randint(1, 3), len(best_result) - 1)
            perturb = list(best_result)
            for _ in range(n_remove):
                if perturb:
                    idx = random.randint(0, len(perturb) - 1)
                    perturb.pop(idx)
            # Recompute sums
            n = len(perturb)
            sums = set()
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += perturb[i]
                sums.add(s)
        else:
            perturb = list(best_result)
            sums = set()

        # Run simulated annealing
        result = simulated_annealing(perturb, sums, deadline)
        if len(result) > len(best_result):
            best_result = result

    return [int(x) for x in best_result]
