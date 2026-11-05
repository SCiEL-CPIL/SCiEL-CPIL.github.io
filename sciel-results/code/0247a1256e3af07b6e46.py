"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _compute_sums(elems):
        """Compute all pair sums including a+a."""
        s = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s.add(elems[i] + elems[j])
        return s
    
    def _compute_diffs(elems):
        """Compute all nonzero differences within the set."""
        d = set()
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                diff = abs(elems[i] - elems[j])
                if diff > 0:
                    d.add(diff)
        return d
    
    def _can_add_to_sidon(x, elems, sums):
        """Check if x can be added to maintain Sidon property."""
        for a in elems:
            if a + x in sums:
                return False
        if 2 * x in sums:
            return False
        return True
    
    def _greedy_build_sidon(order, excluded_diffs, deadline):
        """
        Greedily build a Sidon set, avoiding differences in excluded_diffs.
        Returns the constructed set.
        """
        elems = []
        sums = set()
        diffs = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check Sidon constraint
            if not _can_add_to_sidon(x, elems, sums):
                continue
            
            # Check difference constraint: new diffs must not conflict with excluded_diffs
            new_diffs_conflict = False
            for a in elems:
                d = abs(a - x)
                if d > 0 and d in excluded_diffs:
                    new_diffs_conflict = True
                    break
            
            if new_diffs_conflict:
                continue
            
            # Add x
            for a in elems:
                sums.add(a + x)
                d = abs(a - x)
                if d > 0:
                    diffs.add(d)
            sums.add(2 * x)
            elems.append(x)
        
        return sorted(elems)
    
    class State:
        def __init__(self, A, B):
            self.A = sorted(A)
            self.B = sorted(B)
            self.sumsA = _compute_sums(self.A)
            self.sumsB = _compute_sums(self.B)
            self.diffsA = _compute_diffs(self.A)
            self.diffsB = _compute_diffs(self.B)
            self.size = len(self.A) + len(self.B)

        def can_add_to_A(self, x):
            if x in self.A: return False
            # Check Sidon A
            for a in self.A:
                if a + x in self.sumsA:
                    return False
            if 2*x in self.sumsA:
                return False
            # Check Diff Disjoint
            for a in self.A:
                d = abs(a - x)
                if d > 0 and d in self.diffsB:
                    return False
            return True

        def add_to_A(self, x):
            for a in self.A:
                self.sumsA.add(a + x)
                d = abs(a - x)
                if d > 0:
                    self.diffsA.add(d)
            self.sumsA.add(2*x)
            self.A.append(x)
            self.size += 1

        def remove_from_A(self, x):
            self.A.remove(x)
            self.sumsA.discard(2*x)
            # Rebuild sums and diffs is safer for removal, but we can try to remove pairs
            # Actually, removing from a set of sums is tricky if duplicates exist? No, sums are unique in Sidon.
            # But wait, sumsA is a set. If a+x was added, we remove it.
            # However, if a+x == b+x' for some other pair? No, Sidon means unique.
            # So we can just remove the specific sums involving x.
            for a in self.A:
                self.sumsA.discard(a + x)
                d = abs(a - x)
                if d > 0:
                    # We cannot simply discard d from diffsA if d is also formed by other pairs in A
                    # So we must rebuild diffsA if we remove.
                    pass
            
            # Rebuilding is O(N^2). For N~30, it's 900 ops. Acceptable if infrequent.
            self.diffsA = _compute_diffs(self.A)
            self.size -= 1

        def can_add_to_B(self, x):
            if x in self.B: return False
            for b in self.B:
                if b + x in self.sumsB:
                    return False
            if 2*x in self.sumsB:
                return False
            for b in self.B:
                d = abs(b - x)
                if d > 0 and d in self.diffsA:
                    return False
            return True

        def add_to_B(self, x):
            for b in self.B:
                self.sumsB.add(b + x)
                d = abs(b - x)
                if d > 0:
                    self.diffsB.add(d)
            self.sumsB.add(2*x)
            self.B.append(x)
            self.size += 1

        def remove_from_B(self, x):
            self.B.remove(x)
            self.sumsB.discard(2*x)
            for b in self.B:
                self.sumsB.discard(b + x)
            self.diffsB = _compute_diffs(self.B)
            self.size -= 1

        def copy(self):
            # Deep copy is expensive, but we only need it for best tracking if we modify in place
            # We will modify in place and keep a separate best snapshot
            return State(list(self.A), list(self.B))

    def _local_search_two_sets(setA, setB, deadline, max_iters=2000):
        state = State(setA, setB)
        best_state = state.copy()
        best_size = state.size
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try adding to A
            if random.random() < 0.25:
                # Pick a candidate not in A
                # To speed up, we can just pick random 1..n
                x = random.randint(1, n)
                if state.can_add_to_A(x):
                    state.add_to_A(x)
                    if state.size > best_size:
                        best_state = state.copy()
                        best_size = state.size
            
            # Try adding to B
            elif random.random() < 0.5:
                x = random.randint(1, n)
                if state.can_add_to_B(x):
                    state.add_to_B(x)
                    if state.size > best_size:
                        best_state = state.copy()
                        best_size = state.size
            
            # Try move from A to B
            elif random.random() < 0.75 and state.A:
                idx = random.randint(0, len(state.A) - 1)
                x = state.A[idx]
                # Remove from A
                state.remove_from_A(x)
                # Try add to B
                if state.can_add_to_B(x):
                    state.add_to_B(x)
                    if state.size > best_size:
                        best_state = state.copy()
                        best_size = state.size
                else:
                    # Revert
                    state.add_to_A(x)
                    # Note: add_to_A assumes x not in A, which is true.
                    # But we need to ensure state is consistent.
                    # remove_from_A rebuilt diffsA. add_to_A adds back sums/diffs.
                    # This should restore state.
            
            # Try move from B to A
            elif state.B:
                idx = random.randint(0, len(state.B) - 1)
                x = state.B[idx]
                state.remove_from_B(x)
                if state.can_add_to_A(x):
                    state.add_to_A(x)
                    if state.size > best_size:
                        best_state = state.copy()
                        best_size = state.size
                else:
                    state.add_to_B(x)
        
        return best_state.A, best_state.B
    
    # Strategy: Build A first with a good structure, then build B avoiding A's differences
    best_sets = ([1], [2])
    best_size = 2
    
    rng = random.Random(42)
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Build A greedily
        order_A = list(range(1, n + 1))
        rng.shuffle(order_A)
        setA = _greedy_build_sidon(order_A, set(), deadline)
        
        if not setA:
            continue
        
        # Compute A's differences
        diffA = _compute_diffs(setA)
        
        # Build B avoiding A's differences
        order_B = list(range(1, n + 1))
        rng.shuffle(order_B)
        setB = _greedy_build_sidon(order_B, diffA, deadline)
        
        if not setB:
            continue
        
        current_size = len(setA) + len(setB)
        
        # Local search to improve
        if current_size > best_size or (current_size >= best_size and restart > 50):
            setA_ls, setB_ls = _local_search_two_sets(setA, setB, deadline, max_iters=2000)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size
        elif current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
