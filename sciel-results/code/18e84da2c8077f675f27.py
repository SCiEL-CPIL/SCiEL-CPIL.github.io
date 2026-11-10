import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def make_mults(A_list):
        mults = np.zeros(257, dtype=np.int32)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                s = ai + A_list[j]
                if s <= 256:
                    mults[s] += 1
        return mults

    A = list(range(64, 129))
    in_A = np.zeros(129, dtype=bool)
    for x in A:
        in_A[x] = True
    mults = make_mults(A)

    def compute_obj(mults_arr):
        peak = int(np.max(mults_arr[128:257]))
        unc = int(np.sum(mults_arr[128:257] == 0))
        if unc > 0:
            return 10000 + unc * 1000 + peak
        return peak

    cur_obj = compute_obj(mults)
    best_obj = cur_obj
    best_A = A[:]

    temp = 15.0
    cooling = 0.9999

    while time.time() < deadline:
        r = random.random()

        if r < 0.4 and len(A) > 12:
            idx = random.randint(0, len(A) - 1)
            x = A[idx]
            saved_A = A[:]
            saved_in_A = in_A.copy()
            saved_mults = mults.copy()

            for y in A:
                s = x + y
                if s <= 256:
                    mults[s] -= 1
            in_A[x] = False
            A.pop(idx)

        elif r < 0.75:
            x = random.randint(0, 128)
            if in_A[x]:
                temp *= cooling
                continue
            saved_A = A[:]
            saved_in_A = in_A.copy()
            saved_mults = mults.copy()

            for y in A:
                s = x + y
                if s <= 256:
                    mults[s] += 1
            s = 2 * x
            if s <= 256:
                mults[s] += 1
            in_A[x] = True
            A.append(x)

        elif r < 0.9 and len(A) > 12:
            idx = random.randint(0, len(A) - 1)
            x = A[idx]
            y = random.randint(0, 128)
            if in_A[y]:
                temp *= cooling
                continue
            saved_A = A[:]
            saved_in_A = in_A.copy()
            saved_mults = mults.copy()

            for z in A:
                s = x + z
                if s <= 256:
                    mults[s] -= 1
            for z in A:
                if z != x:
                    s = y + z
                    if s <= 256:
                        mults[s] += 1
            s = 2 * y
            if s <= 256:
                mults[s] += 1
            in_A[x] = False
            in_A[y] = True
            A[idx] = y

        else:
            temp *= cooling
            continue

        new_obj = compute_obj(mults)
        delta = new_obj - cur_obj

        if delta < 0 or (temp > 0.01 and random.random() < np.exp(-delta / temp)):
            cur_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = A[:]
        else:
            A = saved_A
            in_A = saved_in_A
            mults = saved_mults

        temp *= cooling
        if temp < 0.1:
            temp = 0.1

    final_mults = make_mults(best_A)
    final_unc = int(np.sum(final_mults[128:257] == 0))
    if final_unc == 0:
        return best_A
    return list(range(64, 129))
