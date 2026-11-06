def calc_makespan(perm):
    completion = np.zeros(n_machines)
    for j in perm:
        job_times = processing_times[j]
        for m in range(n_machines):
            if m == 0:
                st = completion[0]
            else:
                st = completion[m]
                prev = completion[m-1]
                if prev > st:
                    st = prev
            completion[m] = st + job_times[m]
    return completion[-1]
