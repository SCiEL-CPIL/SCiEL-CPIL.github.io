<<<<<<< SEARCH
            intersections = np.bitwise_and(masks_np, uncovered_np)
            # unpackbits requires uint8 input
            bits = np.unpackbits(intersections.view(np.uint8), axis=1)
            gains = bits.sum(axis=1)

            # Mask out used blocks
            gains[used] = -1
=======
            intersections = np.bitwise_and(masks_np, uncovered_np)
            # unpackbits requires uint8 input
            bits = np.unpackbits(intersections.view(np.uint8), axis=1)
            gains = bits.sum(axis=1).astype(np.int32)

            # Mask out used blocks
            gains[used] = -1
>>>>>>> REPLACE
