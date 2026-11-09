"""Improved: degree-aware greedy with multi-pass polish and balanced set growth."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55  # safety margin before 60s kill

    def can_add_to_set(x, S, sums_S, all_diffs):
        """Check if x can be added to set S with given sums and global diffs."""
        if not S:
            return True
        for s in S:
            d = abs(x - s)
            if d in all_diffs:
                return False
        for s in S:
            if x + s in sums_S:
                return False
        if 2 * x in sums_S:
            return False
        return True

    def add_to_set(x, S, sums_S, all_diffs):
        """Add x to set S, updating sums and diffs."""
        for s in S:
            all_diffs.add(abs(x - s))
            sums_S.add(x + s)
        sums_S.add(2 * x)
        S.append(x)

    def greedy_balanced(seed=None):
        """Greedy that balances growth between A and B using degree heuristics."""
        if seed is not None:
            random.seed(seed)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        all_diffs = set()
        
        # Precompute conflict degrees for initial ordering
        candidates = list(range(1, n + 1))
        
        for _ in range(len(candidates)):
            # Pick the candidate with minimum conflict (most flexible)
            best_x = None
            best_score = float('inf')
            best_set_choice = None
            
            for x in candidates:
                # Score: number of conflicts if added to A or B
                conflicts_A = 0
                for a in A:
                    if abs(x - a) in all_diffs:
                        conflicts_A += 1
                    if x + a in sums_A:
                        conflicts_A += 1
                if 2 * x in sums_A:
                    conflicts_A += 1
                
                conflicts_B = 0
                for b in B:
                    if abs(x - b) in all_diffs:
                        conflicts_B += 1
                    if x + b in sums_B:
                        conflicts_B += 1
                if 2 * x in sums_B:
                    conflicts_B += 1
                
                # Prefer adding to the smaller set if possible
                if len(A) <= len(B):
                    primary, secondary = 'A', 'B'
                    primary_conflicts, secondary_conflicts = conflicts_A, conflicts_B
                else:
                    primary, secondary = 'B', 'A'
                    primary_conflicts, secondary_conflicts = conflicts_B, conflicts_A
                
                # Choose the option with fewer conflicts
                if primary_conflicts < secondary_conflicts:
                    score = primary_conflicts
                    choice = primary
                else:
                    score = secondary_conflicts
                    choice = secondary
                
                # Add small random perturbation for diversity
                score += random.random() * 0.1
                
                if score < best_score:
                    best_score = score
                    best_x = x
                    best_set_choice = choice
            
            if best_x is None:
                break
            
            candidates.remove(best_x)
            
            # Add to chosen set
            if best_set_choice == 'A' and len(A) < max_size:
                if can_add_to_set(best_x, A, sums_A, all_diffs):
                    add_to_set(best_x, A, sums_A, all_diffs)
            elif best_set_choice == 'B' and len(B) < max_size:
                if can_add_to_set(best_x, B, sums_B, all_diffs):
                    add_to_set(best_x, B, sums_B, all_diffs)
            else:
                # Try the other set
                if best_set_choice == 'A':
                    if len(B) < max_size and can_add_to_set(best_x, B, sums_B, all_diffs):
                        add_to_set(best_x, B, sums_B, all_diffs)
                else:
                    if len(A) < max_size and can_add_to_set(best_x, A, sums_A, all_diffs):
                        add_to_set(best_x, A, sums_A, all_diffs)
        
        return A, B

    def greedy_random(seed=None):
        """Original random-order greedy."""
        if seed is not None:
            random.seed(seed)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        all_diffs = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try A first
            if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                add_to_set(x, A, sums_A, all_diffs)
                continue
            # Try B
            if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                add_to_set(x, B, sums_B, all_diffs)
        
        return A, B

    def greedy_interleaved(seed=None):
        """Interleave A and B additions to balance difference usage."""
        if seed is not None:
            random.seed(seed)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        all_diffs = set()
        
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        add_to_A = True
        for x in candidates:
            if add_to_A:
                if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                    add_to_set(x, A, sums_A, all_diffs)
                    add_to_A = False
                    continue
                if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                    add_to_set(x, B, sums_B, all_diffs)
                    add_to_A = True
                    continue
            else:
                if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                    add_to_set(x, B, sums_B, all_diffs)
                    add_to_A = True
                    continue
                if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                    add_to_set(x, A, sums_A, all_diffs)
                    add_to_A = False
                    continue
        
        return A, B

    def polish_multi_pass(A, B, time_limit=3.0):
        """Multi-pass polish: try inserting, then swapping, then re-inserting."""
        sums_A = set()
        sums_B = set()
        all_diffs = set()

        for a in A:
            for a2 in A:
                sums_A.add(a + a2)
        for b in B:
            for b2 in B:
                sums_B.add(b + b2)
        for a in A:
            for a2 in A:
                if a != a2:
                    all_diffs.add(abs(a - a2))
        for b in B:
            for b2 in B:
                if b != b2:
                    all_diffs.add(abs(b - b2))

        start = time.time()
        
        # Pass 1: Insert remaining elements
        used = set(A) | set(B)
        remaining = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(remaining)
        
        for x in remaining:
            if time.time() - start > time_limit:
                break
            if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                add_to_set(x, A, sums_A, all_diffs)
                continue
            if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                add_to_set(x, B, sums_B, all_diffs)
        
        # Pass 2: Try swapping elements to free up space
        if time.time() - start < time_limit * 0.7:
            used = set(A) | set(B)
            remaining = [x for x in range(1, n + 1) if x not in used]
            random.shuffle(remaining)
            
            for x in remaining:
                if time.time() - start > time_limit:
                    break
                if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                    add_to_set(x, A, sums_A, all_diffs)
                    continue
                if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                    add_to_set(x, B, sums_B, all_diffs)
        
        return A, B

    def local_search(A, B, time_limit=2.0):
        """Try removing and re-adding elements to improve packing."""
        sums_A = set()
        sums_B = set()
        all_diffs = set()

        for a in A:
            for a2 in A:
                sums_A.add(a + a2)
        for b in B:
            for b2 in B:
                sums_B.add(b + b2)
        for a in A:
            for a2 in A:
                if a != a2:
                    all_diffs.add(abs(a - a2))
        for b in B:
            for b2 in B:
                if b != b2:
                    all_diffs.add(abs(b - b2))

        start = time.time()
        best_size = len(A) + len(B)
        best_A = A[:]
        best_B = B[:]
        
        # Try removing one element and re-adding two
        for _ in range(20):
            if time.time() - start > time_limit:
                break
            
            A = best_A[:]
            B = best_B[:]
            sums_A = set()
            sums_B = set()
            all_diffs = set()
            
            for a in A:
                for a2 in A:
                    sums_A.add(a + a2)
            for b in B:
                for b2 in B:
                    sums_B.add(b + b2)
            for a in A:
                for a2 in A:
                    if a != a2:
                        all_diffs.add(abs(a - a2))
            for b in B:
                for b2 in B:
                    if b != b2:
                        all_diffs.add(abs(b - b2))
            
            # Remove a random element
            if A:
                idx = random.randint(0, len(A) - 1)
                A.pop(idx)
            elif B:
                idx = random.randint(0, len(B) - 1)
                B.pop(idx)
            
            # Rebuild sums and diffs
            sums_A = set()
            sums_B = set()
            all_diffs = set()
            for a in A:
                for a2 in A:
                    sums_A.add(a + a2)
            for b in B:
                for b2 in B:
                    sums_B.add(b + b2)
            for a in A:
                for a2 in A:
                    if a != a2:
                        all_diffs.add(abs(a - a2))
            for b in B:
                for b2 in B:
                    if b != b2:
                        all_diffs.add(abs(b - b2))
            
            # Try to add two new elements
            used = set(A) | set(B)
            candidates = [x for x in range(1, n + 1) if x not in used]
            random.shuffle(candidates)
            
            added = 0
            for x in candidates:
                if added >= 2:
                    break
                if len(A) < max_size and can_add_to_set(x, A, sums_A, all_diffs):
                    add_to_set(x, A, sums_A, all_diffs)
                    added += 1
                    continue
                if len(B) < max_size and can_add_to_set(x, B, sums_B, all_diffs):
                    add_to_set(x, B, sums_B, all_diffs)
                    added += 1
            
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = A[:]
                best_B = B[:]
        
        return best_A, best_B

    best_A = []
    best_B = []
    best_size = 0
    seed_counter = 0

    # Phase 1: balanced greedy restarts
    while time.time() < deadline - 15:
        A, B = greedy_balanced(seed=seed_counter)
        A, B = polish_multi_pass(A, B, time_limit=1.5)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Phase 2: random-order greedy
    while time.time() < deadline - 8:
        A, B = greedy_random(seed=seed_counter)
        A, B = polish_multi_pass(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Phase 3: interleaved greedy
    while time.time() < deadline - 3:
        A, B = greedy_interleaved(seed=seed_counter)
        A, B = polish_multi_pass(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Final: local search on best solution
    if best_size > 0:
        best_A, best_B = local_search(best_A, best_B, time_limit=2.0)
        best_A, best_B = polish_multi_pass(best_A, best_B, time_limit=1.0)

    return [best_A, best_B]
