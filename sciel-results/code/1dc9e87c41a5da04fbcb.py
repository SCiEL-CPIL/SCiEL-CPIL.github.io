import time
import random
import math

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    n = 256
    candidates = list(range(1, n + 1))
    
    best_size = 0
    best_A = []
    best_B = []
    
    class State:
        def __init__(self, A, B):
            self.A = A
            self.B = B
            self.sums_A = set()
            self.sums_B = set()
            self.diffs_A = set()
            self.diffs_B = set()
            self._rebuild()
            
        def _rebuild(self):
            self.sums_A = set()
            self.sums_B = set()
            self.diffs_A = set()
            self.diffs_B = set()
            
            for i in range(len(self.A)):
                for j in range(i, len(self.A)):
                    self.sums_A.add(self.A[i] + self.A[j])
                    if i != j:
                        self.diffs_A.add(abs(self.A[i] - self.A[j]))
            
            for i in range(len(self.B)):
                for j in range(i, len(self.B)):
                    self.sums_B.add(self.B[i] + self.B[j])
                    if i != j:
                        self.diffs_B.add(abs(self.B[i] - self.B[j]))
                        
        def can_add(self, x, target, other_diffs):
            """Check if x can be added to target set."""
            if target is self.A:
                sums = self.sums_A
            else:
                sums = self.sums_B
                
            if 2 * x in sums:
                return False
            for a in target:
                if x + a in sums:
                    return False
                d = abs(x - a)
                if d in other_diffs:
                    return False
            return True

        def add(self, x, target):
            """Add x to target set and update state."""
            if target is self.A:
                sums = self.sums_A
                diffs = self.diffs_A
                other_diffs = self.diffs_B
            else:
                sums = self.sums_B
                diffs = self.diffs_B
                other_diffs = self.diffs_A
                
            sums.add(2 * x)
            for a in target:
                sums.add(x + a)
                diffs.add(abs(x - a))
            target.append(x)
            
        def remove(self, x, target):
            """Remove x from target set and update state."""
            if target is self.A:
                sums = self.sums_A
                diffs = self.diffs_A
            else:
                sums = self.sums_B
                diffs = self.diffs_B
                
            target.remove(x)
            self._rebuild()

    def greedy_construct(order, prefer_A_first=True):
        """Greedy construction with given order."""
        A = []
        B = []
        state = State(A, B)
        
        for x in order:
            if time.time() - start_time > time_limit:
                break
            
            if prefer_A_first:
                if state.can_add(x, A, state.diffs_B):
                    state.add(x, A)
                elif state.can_add(x, B, state.diffs_A):
                    state.add(x, B)
            else:
                if state.can_add(x, B, state.diffs_A):
                    state.add(x, B)
                elif state.can_add(x, A, state.diffs_B):
                    state.add(x, A)
                    
        return A, B

    def local_search(A, B, max_iter=2000):
        """
        Local search using add, move, and swap moves.
        """
        A = A[:]
        B = B[:]
        state = State(A, B)
        
        no_improve_count = 0
        max_no_improve = 100
        
        for iteration in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            
            # 1. Try to add any element to A or B
            added = False
            for x in range(1, n + 1):
                if x in A or x in B:
                    continue
                if state.can_add(x, A, state.diffs_B):
                    state.add(x, A)
                    improved = True
                    added = True
                    break
                elif state.can_add(x, B, state.diffs_A):
                    state.add(x, B)
                    improved = True
                    added = True
                    break
            
            if added:
                no_improve_count = 0
                continue
            
            # 2. No add possible. Try swap moves to escape local optimum.
            # Try: remove a from A, add b to A (or B)
            swapped = False
            
            # Try swapping within A: remove a, add b to A
            if A:
                a = random.choice(A)
                A.remove(a)
                state._rebuild()
                
                # Try to add a new element to A
                found = False
                for b in range(1, n + 1):
                    if b in A or b in B:
                        continue
                    if state.can_add(b, A, state.diffs_B):
                        state.add(b, A)
                        swapped = True
                        found = True
                        break
                
                if not found:
                    # Revert
                    A.append(a)
                    state._rebuild()
                    no_improve_count += 1
            elif B:
                b = random.choice(B)
                B.remove(b)
                state._rebuild()
                
                # Try to add a new element to B
                found = False
                for a in range(1, n + 1):
                    if a in A or a in B:
                        continue
                    if state.can_add(a, B, state.diffs_A):
                        state.add(a, B)
                        swapped = True
                        found = True
                        break
                
                if not found:
                    B.append(b)
                    state._rebuild()
                    no_improve_count += 1
            else:
                no_improve_count += 1
            
            if swapped:
                no_improve_count = 0
            elif no_improve_count > max_no_improve:
                break
                
        return A, B

    def create_biased_order(bias_strength=0.3):
        """Create a biased ordering with smaller numbers preferred."""
        order = list(range(1, n + 1))
        for i in range(len(order) - 1):
            if random.random() < bias_strength * (1 - i / len(order)):
                j = random.randint(i, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        return order

    # Main search loop
    restarts = 0
    while time.time() - start_time < time_limit:
        # Vary bias strength for diversity
        bias = 0.2 + random.random() * 0.3
        order = create_biased_order(bias)
        
        prefer_A = random.random() < 0.5
        A, B = greedy_construct(order, prefer_A)
        
        # Local improvement pass
        A, B = local_search(A, B)
        
        current_size = len(A) + len(B)
        if current_size > best_size:
            best_size = current_size
            best_A = A[:]
            best_B = B[:]
        
        restarts += 1
        
        # If we've found a good solution, do more careful refinement
        if best_size >= 28 and restarts % 3 == 0:
            # Try to improve the best solution found so far
            A, B = local_search(best_A, best_B, max_iter=3000)
            if len(A) + len(B) > best_size:
                best_size = len(A) + len(B)
                best_A = A[:]
                best_B = B[:]

    if best_size == 0:
        return [[1], [2]]
        
    return [best_A, best_B]
