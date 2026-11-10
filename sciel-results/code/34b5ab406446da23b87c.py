import numpy as np

def kissing_number11() -> np.ndarray:
    d = 11
    vectors = []
    
    # Generate all vectors with exactly three non-zero entries, which are +/- 1
    # C(11, 3) * 8
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            vec = np.zeros(d, dtype=np.int64)
                            vec[i] = s1
                            vec[j] = s2
                            vec[k] = s3
                            # Check against existing
                            valid = True
                            for v in vectors:
                                diff = vec - v
                                if np.dot(diff, diff) < 3:
                                    valid = False
                                    break
                            if valid:
                                vectors.append(vec)
                    
    return np.array(vectors)
