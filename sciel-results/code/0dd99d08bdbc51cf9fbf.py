<<<<<<< SEARCH
    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_rounded = round(x)
                if abs(x - x_rounded) < 1e-9:
                    result[-1].append(int(x_rounded))
                else:
                    result[-1].append(float(x))
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
=======
    def to_list(mat):
        result = []
        for row in mat:
            result.append([])
            for x in row:
                x_float = float(x)
                # Check if integer
                x_int = round(x_float)
                if abs(x_float - x_int) < 1e-9:
                    result[-1].append(int(x_int))
                else:
                    # Check if half-integer
                    x_half = round(x_float * 2) / 2.0
                    if abs(x_float - x_half) < 1e-9:
                        result[-1].append(x_half)
                    else:
                        result[-1].append(x_float)
        return result

    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
>>>>>>> REPLACE
