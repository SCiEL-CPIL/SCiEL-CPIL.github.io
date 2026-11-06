import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0
    n = 256
    max_size = 64

    def compute_sums(A):
        sums = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
        return sums

    def compute_diffs(A):
        diffs = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs.add(A[i] - A[j])
        return diffs

    def is_valid(A, B):
        if len(A) > max_size or len(B) > max_size:
            return False
        sums_A = compute_sums(A)
        if len(sums_A) != len(A)*(len(A)+1)//2:
            return False
        sums_B = compute_sums(B)
        if len(sums_B) != len(B)*(len(B)+1)//2:
            return False
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)
        if diffs_A & diffs_B:
            return False
        return True

    def can_add_to_A(x, A, sums_A, diffs_B):
        if len(A) >= max_size or x in A:
            return False
        for a in A:
            if (x+a) in sums_A:
                return False
        if (2*x) in sums_A:
            return False
        for a in A:
            if (x-a) in diffs_B or (a-x) in diffs_B:
                return False
        return True

    def can_add_to_B(x, B, sums_B, diffs_A):
        if len(B) >= max_size or x in B:
            return False
        for b in B:
            if (x+b) in sums_B:
                return False
        if (2*x) in sums_B:
            return False
        for b in B:
            if (x-b) in diffs_A or (b-x) in diffs_A:
                return False
        return True

    def run_greedy(order):
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            if can_add_to_A(x, A, sums_A, diffs_B):
                for a in A:
                    sums_A.add(x+a)
                    diffs_A.add(x-a)
                    diffs_A.add(a-x)
                sums_A.add(2*x)
                A.append(x)
            elif can_add_to_B(x, B, sums_B, diffs_A):
                for b in B:
                    sums_B.add(x+b)
                    diffs_B.add(x-b)
                    diffs_B.add(b-x)
                sums_B.add(2*x)
                B.append(x)

        return A, B

    def local_search(A, B, deadline):
        A = list(A)
        B = list(B)
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        diffs_A = compute_diffs(A)
        diffs_B = compute_diffs(B)

        best_size = len(A) + len(B)
        best_A, best_B = A[:], B[:]
        all_elements = list(range(1, n+1))

        while time.time() < deadline:
            improved = False

            # Try adding elements
            for x in all_elements:
                if x not in A and can_add_to_A(x, A, sums_A, diffs_B):
                    for a in A:
                        sums_A.add(x+a)
                        diffs_A.add(x-a)
                        diffs_A.add(a-x)
                    sums_A.add(2*x)
                    A.append(x)
                    best_size = len(A) + len(B)
                    best_A, best_B = A[:], B[:]
                    improved = True
                elif x not in B and can_add_to_B(x, B, sums_B, diffs_A):
                    for b in B:
                        sums_B.add(x+b)
                        diffs_B.add(x-b)
                        diffs_B.add(b-x)
                    sums_B.add(2*x)
                    B.append(x)
                    best_size = len(A) + len(B)
                    best_A, best_B = A[:], B[:]
                    improved = True

            # Try moving elements between sets
            for x in list(A):
                # Try moving x from A to B
                if x not in B:
                    # Check if we can remove x from A and add to B
                    # First check if B can accept x
                    if can_add_to_B(x, B, sums_B, diffs_A):
                        # Remove x from A
                        A.remove(x)
                        sums_A = compute_sums(A)
                        diffs_A = compute_diffs(A)
                        # Add x to B
                        for b in B:
                            sums_B.add(x+b)
                            diffs_B.add(x-b)
                            diffs_B.add(b-x)
                        sums_B.add(2*x)
                        B.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True

            for x in list(B):
                if x not in A:
                    if can_add_to_A(x, A, sums_A, diffs_B):
                        B.remove(x)
                        sums_B = compute_sums(B)
                        diffs_B = compute_diffs(B)
                        for a in A:
                            sums_A.add(x+a)
                            diffs_A.add(x-a)
                            diffs_A.add(a-x)
                        sums_A.add(2*x)
                        A.append(x)
                        best_size = len(A) + len(B)
                        best_A, best_B = A[:], B[:]
                        improved = True

            if not improved:
                break

        return best_A, best_B

    best_A = []
    best_B = []
    best_total = 0

    for restart in range(300):
        if time.time() - start_time > time_limit:
            break
        random.seed(restart * 31 + 17)
        order = list(range(1, n+1))
        random.shuffle(order)

        A, B = run_greedy(order)
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A, best_B = A[:], B[:]

        A2, B2 = local_search(A, B, time_limit)
        total2 = len(A2) + len(B2)
        if total2 > best_total:
            best_total = total2
            best_A, best_B = A2[:], B2[:]

    return [best_A, best_B]
