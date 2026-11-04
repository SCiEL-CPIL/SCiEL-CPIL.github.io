    def to_rational_array(arr, tol=1e-6):
        res = []
        for row in arr:
            new_row = []
            for val in row:
                # Try nearest integer
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    new_row.append(nearest_int)
                else:
                    # Try nearest half-integer
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        new_row.append(nearest_half)
                    else:
                        return None # Failed to rationalize
            res.append(new_row)
        return res
