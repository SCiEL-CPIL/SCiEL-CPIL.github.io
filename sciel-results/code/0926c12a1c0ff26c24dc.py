import time, random, math

def construct(n_rows, k):
    start = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    def row_diffs(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i+1, n):
                s.add(row[j]-ai)
        return s

    # ---- Greedy warm start (valid) ----
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k+1:
                x += 1
                new = {x-a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy()
    diff_counts = {}
    for row in rows:
        for d in row_diffs(row):
            diff_counts[d] = diff_counts.get(d,0)+1
    collision_count = sum(1 for c in diff_counts.values() if c>=2)
    best_rows = [list(r) for r in rows]
    best_max = max(r[-1] for r in best_rows)

    def energy(collision_count, rows):
        return collision_count*1000000 + max(r[-1] for r in rows)

    # incremental apply
    def apply_move(rows, r_idx, new_row, diff_counts, collision_count):
        old_row = rows[r_idx]
        old_diffs = row_diffs(old_row)
        new_diffs = row_diffs(new_row)
        for d in old_diffs:
            c = diff_counts[d]
            if c==2: collision_count-=1
            c-=1
            if c==0: del diff_counts[d]
            else: diff_counts[d]=c
        for d in new_diffs:
            c = diff_counts.get(d,0)
            if c==1: collision_count+=1
            diff_counts[d]=c+1
        rows[r_idx]=new_row
        return collision_count

    def valid_row(row):
        if len(row)!=k+1 or row[0]!=0: return False
        for i in range(1,len(row)):
            if row[i]<=row[i-1]: return False
        # intra-row distinct diffs
        if len(row_diffs(row)) != (k+1)*k//2: return False
        return True

    best_energy = energy(collision_count, rows)

    T = 500.0
    T_min = 0.5
    cool = 0.9999
    rng = random.Random(12345)

    while time.time() < DEADLINE:
        # pick a row, bias toward larger
        weights = [r[-1]+1 for r in rows]
        total = sum(weights)
        pick = rng.random()*total
        acc=0; r_idx=0
        for i,w in enumerate(weights):
            acc+=w
            if pick<=acc: r_idx=i; break
        cur_row = rows[r_idx]
        cur_energy = energy(collision_count, rows)

        # choose move type
        if rng.random() < 0.35:
            # rebuild row greedily with randomization
            used_others = set(diff_counts.keys())
            # remove this row's own from consideration? we rebuild avoiding all used including others
            row=[0]; x=0
            # random start bias
            while len(row)<k+1:
                x += 1
                new={x-a for a in row}
                if not (new & used_others):
                    row.append(x); 
                else:
                    continue
            # this may exceed; just accept as candidate
            cand = row
        else:
            # point move
            i = rng.randint(1, k)  # interior index 1..k
            lo = cur_row[i-1]+1
            hi = cur_row[i+1]-1 if i<k else max(best_max, cur_row[-1])+3
            if hi<lo: hi=lo
            v = rng.randint(lo,hi)
            cand = list(cur_row); cand[i]=v; cand.sort()
            if not valid_row(cand):
                continue

        new_cc = apply_move(rows, r_idx, cand, diff_counts, collision_count)
        new_energy = energy(new_cc, rows)
        delta = new_energy - cur_energy
        if delta <= 0 or rng.random() < math.exp(-delta/T):
            if new_energy < best_energy:
                best_energy = new_energy
                if new_cc==0:
                    bm = max(r[-1] for r in rows)
                    if bm < best_max:
                        best_max = bm
                        best_rows=[list(r) for r in rows]
        else:
            # revert
            new_cc = apply_move(rows, r_idx, cur_row, diff_counts, new_cc)
        T = max(T_min, T*cool)

    # ensure best_rows valid & return
    # final validation
    used=set()
    for row in best_rows:
        ds=row_diffs(row)
        if ds & used: 
            # fallback to greedy
            return greedy()
        used|=ds
    return [list(r) for r in best_rows]
