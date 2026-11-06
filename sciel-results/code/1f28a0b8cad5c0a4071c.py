        for x in list(smooth):
            val = x
            while val * p <= limit:
                val *= p
                new_smooth.add(val)
